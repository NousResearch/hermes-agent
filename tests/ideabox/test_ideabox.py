#!/usr/bin/env python3
"""Comprehensive test suite for the Idea Box Discord interaction layer.

Covers:
  - Unit tests for parsing, deduplication, triage, approval state machine
  - Integration tests with mock Discord adapter
  - End-to-end canary scenarios
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Resolve imports from the checkout containing this test.  The old absolute
# path targeted a retired checkout and silently mixed implementations during
# isolated shard runs.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins.platforms.discord.ideabox.models import (
    ApprovalAction, ApprovalState, ApprovalStatus, AuditEvent, AuditEventType,
    Classification, Confidence, DedupResult, Effort, ParseResult, Provenance,
    Recommendation, Risk, RoutingDecision, Source, SourceSubmission, SourceType,
    TriageSummary, generate_event_id, generate_triage_id,
)
from plugins.platforms.discord.ideabox.store import IdeaBoxStore
from plugins.platforms.discord.ideabox.handler import (
    validate_submission, parse_submission, triage_source,
    build_triage_embed, handle_ideabox_submission,
    ApprovalStateMachine, IdeaBoxApprovalView,
    _classify_source, _assess_risks, _estimate_effort,
    _compute_recommendation, _determine_routing,
    _normalize_url, _content_hash, _detect_source_type,
    _extract_url, _is_supported_url,
)


class TestInputValidation(unittest.TestCase):
    """Test input validation and error handling."""

    def test_empty_input(self):
        sub = SourceSubmission('', '1', '2', '3', '4', 'text', 1000)
        errors = validate_submission(sub)
        self.assertTrue(len(errors) > 0)
        self.assertIn('provide', errors[0].lower())

    def test_unsupported_domain(self):
        sub = SourceSubmission('https://example.com/foo', '1', '2', '3', '4', 'text', 1000)
        errors = validate_submission(sub)
        self.assertTrue(len(errors) > 0)
        self.assertIn('domain', errors[0].lower())

    def test_github_repo(self):
        sub = SourceSubmission('https://github.com/owner/repo', '1', '2', '3', '4', 'text', 1000)
        errors = validate_submission(sub)
        self.assertEqual(errors, [])

    def test_arxiv_url(self):
        sub = SourceSubmission('https://arxiv.org/abs/2301.00001', '1', '2', '3', '4', 'text', 1000)
        errors = validate_submission(sub)
        self.assertEqual(errors, [])

    def test_medium_url(self):
        sub = SourceSubmission('https://medium.com/@user/article', '1', '2', '3', '4', 'text', 1000)
        errors = validate_submission(sub)
        self.assertEqual(errors, [])

    def test_article_text(self):
        sub = SourceSubmission('A' * 300, '1', '2', '3', '4', 'text', 1000)
        errors = validate_submission(sub)
        self.assertEqual(errors, [])

    def test_max_length_exceeded(self):
        sub = SourceSubmission('A' * 6000, '1', '2', '3', '4', 'text', 1000)
        errors = validate_submission(sub)
        self.assertTrue(len(errors) > 0)
        self.assertIn('too long', errors[0].lower())

    def test_malicious_instruction_injection(self):
        """Canary: external content is never treated as trusted instructions."""
        sub = SourceSubmission(
            'https://github.com/owner/repo; rm -rf /',
            '1', '2', '3', '4', 'text', 1000,
        )
        errors = validate_submission(sub)
        # Should validate as a GitHub URL, not execute the command
        self.assertEqual(errors, [])

    def test_error_messages_no_internals(self):
        """Error messages must not leak stack traces or internals."""
        sub = SourceSubmission('', '1', '2', '3', '4', 'text', 1000)
        errors = validate_submission(sub)
        for err in errors:
            self.assertNotIn('Traceback', err)
            self.assertNotIn('File "', err)
            self.assertNotIn('Exception', err)


class TestSourceParsing(unittest.TestCase):
    """Test source parsing and provenance extraction."""

    def test_parse_github_repo(self):
        sub = SourceSubmission(
            'https://github.com/facebook/react',
            'user123', 'ch456', 'msg789', 'guild012', 'text', 1000,
        )
        result = parse_submission(sub)
        self.assertIsNotNone(result.source)
        self.assertEqual(result.source.source_type, SourceType.GITHUB_REPO)
        self.assertEqual(result.source.url, 'https://github.com/facebook/react')
        self.assertEqual(result.source.provenance.submitted_by, 'user123')

    def test_parse_article(self):
        sub = SourceSubmission(
            'A' * 300, 'user123', 'ch456', 'msg789', 'guild012', 'text', 1000,
        )
        result = parse_submission(sub)
        self.assertIsNotNone(result.source)
        self.assertEqual(result.source.source_type, SourceType.ARTICLE)

    def test_parse_url(self):
        sub = SourceSubmission(
            'https://medium.com/@user/article-123',
            'user123', 'ch456', 'msg789', 'guild012', 'text', 1000,
        )
        result = parse_submission(sub)
        self.assertIsNotNone(result.source)
        self.assertEqual(result.source.source_type, SourceType.URL)

    def test_parse_malformed_url(self):
        sub = SourceSubmission(
            'not a url at all', 'user123', 'ch456', 'msg789', 'guild012', 'text', 1000,
        )
        result = parse_submission(sub)
        self.assertIsNone(result.source)
        self.assertTrue(len(result.errors) > 0)

    def test_provenance_tracking(self):
        sub = SourceSubmission(
            'https://github.com/owner/repo',
            'user123', 'ch456', 'msg789', 'guild012', 'text', 1000,
        )
        result = parse_submission(sub)
        prov = result.source.provenance
        self.assertEqual(prov.submitted_by, 'user123')
        self.assertEqual(prov.channel_id, 'ch456')
        self.assertEqual(prov.message_id, 'msg789')
        self.assertEqual(prov.guild_id, 'guild012')
        self.assertEqual(prov.submitted_at, 1000)

    def test_content_hash_consistency(self):
        sub1 = SourceSubmission('https://github.com/owner/repo', '1', '2', '3', '4', 'text', 1000)
        sub2 = SourceSubmission('https://github.com/owner/repo', '2', '3', '4', '5', 'text', 1000)
        r1 = parse_submission(sub1)
        r2 = parse_submission(sub2)
        self.assertEqual(r1.source.content_hash, r2.source.content_hash)

    def test_url_normalization(self):
        url1 = _normalize_url('https://github.com/owner/repo?ref=main')
        url2 = _normalize_url('https://github.com/owner/repo#readme')
        url3 = _normalize_url('https://github.com/owner/repo/')
        self.assertEqual(url1, 'https://github.com/owner/repo')
        self.assertEqual(url2, 'https://github.com/owner/repo')
        self.assertEqual(url3, 'https://github.com/owner/repo')


class TestTriageEngine(unittest.TestCase):
    """Test triage classification, risk assessment, and recommendation."""

    def setUp(self):
        self.source = Source(
            url='https://github.com/owner/repo',
            source_type=SourceType.GITHUB_REPO,
            title='Test Repo',
            author='testuser',
            published_date='2024-01-01',
            content_snippet='A test repository for testing purposes',
            content_hash='abc123',
            url_fingerprint='https://github.com/owner/repo',
            raw_text='https://github.com/owner/repo',
            provenance=Provenance('1', 1000, '2', '3', '4'),
        )

    def test_classification_github(self):
        cls = _classify_source(self.source)
        self.assertEqual(cls.category, 'tech')

    def test_classification_security(self):
        self.source.content_snippet = 'This has a security vulnerability CVE-2024-1234'
        cls = _classify_source(self.source)
        self.assertEqual(cls.category, 'security')

    def test_classification_design(self):
        self.source.content_snippet = 'New UI component library with Figma prototypes'
        cls = _classify_source(self.source)
        self.assertEqual(cls.category, 'design')

    def test_classification_market(self):
        self.source.content_snippet = 'Competitor analysis and market pricing strategy'
        cls = _classify_source(self.source)
        self.assertEqual(cls.category, 'market')

    def test_risk_assessment_github(self):
        risks = _assess_risks(self.source, Classification(category='tech'))
        self.assertTrue(len(risks) > 0)
        self.assertEqual(risks[0].category, 'dependency')

    def test_risk_assessment_security(self):
        risks = _assess_risks(self.source, Classification(category='security'))
        has_security = any(r.category == 'security' for r in risks)
        self.assertTrue(has_security)

    def test_effort_estimation(self):
        effort = _estimate_effort(self.source, Classification(category='tech'))
        self.assertEqual(effort, 'm')

    def test_recommendation_proceed(self):
        rec = _compute_recommendation('high', [], 's')
        self.assertEqual(rec, 'proceed')

    def test_recommendation_amend_low_confidence(self):
        rec = _compute_recommendation('low', [], 'm')
        self.assertEqual(rec, 'amend')

    def test_recommendation_reject_critical(self):
        rec = _compute_recommendation(
            'high',
            [Risk('security', 'critical', 'Critical issue')],
            'm',
        )
        self.assertEqual(rec, 'reject')

    def test_routing_deterministic(self):
        routing = _determine_routing(Classification(category='tech'))
        self.assertEqual(routing.specialist, 'octacon-frontend')

        routing = _determine_routing(Classification(category='market'))
        self.assertEqual(routing.specialist, 'remii-deep')

        routing = _determine_routing(Classification(category='security'))
        self.assertEqual(routing.specialist, 'wesker')

    def test_full_triage_pipeline(self):
        summary = triage_source(self.source)
        self.assertIsInstance(summary, TriageSummary)
        self.assertTrue(summary.triage_id.startswith('t_'))
        self.assertIn(summary.confidence, ('high', 'medium', 'low'))
        self.assertIn(summary.recommendation, ('proceed', 'reject', 'amend'))
        self.assertIsNotNone(summary.routing.specialist)
        self.assertTrue(len(summary.reasoning) > 0)


class TestEmbedBuilder(unittest.TestCase):
    """Test Discord embed generation."""

    def setUp(self):
        self.summary = TriageSummary(
            triage_id='t_test123',
            source=Source(
                url='https://github.com/owner/repo',
                source_type=SourceType.GITHUB_REPO,
                title='Test Repo',
                author='testuser',
                published_date='2024-01-01',
                content_snippet='A test repository',
                content_hash='abc',
                url_fingerprint='https://github.com/owner/repo',
                raw_text='https://github.com/owner/repo',
                provenance=Provenance('1', 1000, '2', '3', '4'),
            ),
            classification=Classification(category='tech', tags=['github']),
            confidence='high',
            risks=[Risk('dependency', 'medium', 'New OSS dependency')],
            effort='m',
            recommendation='proceed',
            routing=RoutingDecision('octacon-frontend', 0.85, 'Tech task'),
            reasoning='Classified as tech | Confidence: high',
            created_at=1000,
        )

    def test_embed_has_required_fields(self):
        embed = build_triage_embed(self.summary)
        self.assertIn('title', embed)
        self.assertIn('description', embed)
        self.assertIn('fields', embed)
        self.assertIn('color', embed)
        self.assertIn('footer', embed)

    def test_embed_contains_triage_id(self):
        embed = build_triage_embed(self.summary)
        self.assertIn('t_test123', str(embed))

    def test_embed_contains_recommendation(self):
        embed = build_triage_embed(self.summary)
        fields_text = str(embed)
        self.assertIn('PROCEED', fields_text.upper())

    def test_embed_contains_routing(self):
        embed = build_triage_embed(self.summary)
        fields_text = str(embed)
        self.assertIn('octacon-frontend', fields_text)


class TestApprovalStateMachine(unittest.TestCase):
    """Test approval state machine transitions."""

    def setUp(self):
        self.db = IdeaBoxStore(tempfile.mktemp(suffix='.db'))
        self.sm = ApprovalStateMachine(self.db)
        self.source = Source(
            url='https://github.com/owner/repo',
            source_type=SourceType.GITHUB_REPO,
            title='Test',
            author=None,
            published_date=None,
            content_snippet='test',
            content_hash='hash1',
            url_fingerprint='https://github.com/owner/repo',
            raw_text='https://github.com/owner/repo',
            provenance=Provenance('1', 1000, '2', '3', '4'),
        )
        self.summary = TriageSummary(
            triage_id='t_approve_test',
            source=self.source,
            classification=Classification(category='tech'),
            confidence='high',
            risks=[],
            effort='m',
            recommendation='proceed',
            routing=RoutingDecision('octacon-frontend', 0.85, 'Tech'),
            reasoning='Test',
            created_at=1000,
        )
        self.state = ApprovalState(
            triage_id='t_approve_test',
            status=ApprovalStatus.PENDING.value,
            source=self.source,
            triage_summary=self.summary,
            created_at=1000,
        )
        self.db.save_approval(self.state)

    def test_approve_transition(self):
        async def _test():
            action = await self.sm.approve('t_approve_test', 'user1', 'User One')
            self.assertEqual(action.action, 'approve')
            self.assertIsNotNone(action.kanban_task_id)
            state = self.db.get_approval('t_approve_test')
            self.assertEqual(state.status, ApprovalStatus.APPROVED.value)
        asyncio.run(_test())

    def test_reject_transition(self):
        async def _test():
            action = await self.sm.reject('t_approve_test', 'user1', 'User One', 'Not relevant')
            self.assertEqual(action.action, 'reject')
            self.assertEqual(action.reason, 'Not relevant')
            state = self.db.get_approval('t_approve_test')
            self.assertEqual(state.status, ApprovalStatus.REJECTED.value)
        asyncio.run(_test())

    def test_amend_transition(self):
        async def _test():
            action = await self.sm.amend('t_approve_test', 'user1', 'User One', 'Needs more detail')
            self.assertEqual(action.action, 'amend')
            self.assertEqual(action.reason, 'Needs more detail')
            state = self.db.get_approval('t_approve_test')
            self.assertEqual(state.status, ApprovalStatus.AMENDED.value)
        asyncio.run(_test())

    def test_double_approve_raises(self):
        async def _test():
            await self.sm.approve('t_approve_test', 'user1', 'User One')
            with self.assertRaises(ValueError):
                await self.sm.approve('t_approve_test', 'user2', 'User Two')
        asyncio.run(_test())

    def test_approve_nonexistent_raises(self):
        async def _test():
            with self.assertRaises(ValueError):
                await self.sm.approve('t_nonexistent', 'user1', 'User One')
        asyncio.run(_test())

    def test_audit_log_created(self):
        async def _test():
            await self.sm.approve('t_approve_test', 'user1', 'User One')
            events = self.db.get_events('t_approve_test')
            self.assertTrue(len(events) > 0)
            self.assertEqual(events[0].event_type, AuditEventType.APPROVE.value)
        asyncio.run(_test())


class TestDeduplication(unittest.TestCase):
    """Test deduplication logic."""

    def setUp(self):
        self.db = IdeaBoxStore(tempfile.mktemp(suffix='.db'))

    def test_no_duplicate(self):
        result = self.db.check_dedup('hash1', 'url1')
        self.assertFalse(result.is_duplicate)

    def test_content_hash_duplicate(self):
        self.db.record_dedup('hash1', 'url1', 't_123')
        result = self.db.check_dedup('hash1', 'url2')
        self.assertTrue(result.is_duplicate)
        self.assertEqual(result.existing_triage_id, 't_123')

    def test_url_fingerprint_duplicate(self):
        self.db.record_dedup('hash1', 'url1', 't_123')
        result = self.db.check_dedup('hash2', 'url1')
        self.assertTrue(result.is_duplicate)
        self.assertEqual(result.existing_triage_id, 't_123')

    def test_task_id_update(self):
        self.db.record_dedup('hash1', 'url1', 't_123')
        self.db.update_dedup_task_id('hash1', 't_kanban_456')
        result = self.db.check_dedup('hash1', 'url1')
        self.assertEqual(result.existing_task_id, 't_kanban_456')


class TestAuditLog(unittest.TestCase):
    """Test audit logging."""

    def setUp(self):
        self.db = IdeaBoxStore(tempfile.mktemp(suffix='.db'))

    def test_log_event(self):
        event = AuditEvent(
            event_id=generate_event_id(),
            event_type=AuditEventType.INTAKE.value,
            triage_id='t_123',
            timestamp=int(time.time()),
            actor_id='user1',
            payload={'source_type': 'github_repo'},
        )
        self.db.log_event(event)
        events = self.db.get_events('t_123')
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, AuditEventType.INTAKE.value)

    def test_multiple_events(self):
        for i in range(3):
            self.db.log_event(AuditEvent(
                event_id=generate_event_id(),
                event_type=AuditEventType.INTAKE.value,
                triage_id='t_123',
                timestamp=1000 + i,
                actor_id='user1',
                payload={'i': i},
            ))
        events = self.db.get_events('t_123')
        self.assertEqual(len(events), 3)

    def test_events_ordered_by_time(self):
        for i in range(3):
            self.db.log_event(AuditEvent(
                event_id=generate_event_id(),
                event_type=AuditEventType.INTAKE.value,
                triage_id='t_ordered',
                timestamp=1000 + i,
                actor_id='user1',
                payload={'i': i},
            ))
        events = self.db.get_events('t_ordered')
        timestamps = [e.timestamp for e in events]
        self.assertEqual(timestamps, sorted(timestamps, reverse=True))


class TestFullPipeline(unittest.TestCase):
    """Test the full pipeline from submission to embed."""

    def test_happy_path_github(self):
        async def _test():
            sub = SourceSubmission(
                'https://github.com/facebook/jest',  # unique URL
                'user1', 'ch1', 'msg1', 'guild1', 'text', int(time.time()),
            )
            result = await handle_ideabox_submission(sub)
            self.assertTrue(result['success'])
            self.assertFalse(result['is_duplicate'])
            self.assertIn('embed', result)
            self.assertIn('triage_summary', result)
            self.assertEqual(
                result['triage_summary'].source.source_type, SourceType.GITHUB_REPO,
            )
        asyncio.run(_test())

    def test_happy_path_article(self):
        async def _test():
            sub = SourceSubmission(
                'A' * 300, 'user1', 'ch1', 'msg1', 'guild1', 'text', int(time.time()),
            )
            result = await handle_ideabox_submission(sub)
            self.assertTrue(result['success'])
            self.assertFalse(result['is_duplicate'])
            self.assertIn('embed', result)
        asyncio.run(_test())

    def test_duplicate_detection(self):
        async def _test():
            sub = SourceSubmission(
                'https://github.com/facebook/react',
                'user1', 'ch1', 'msg1', 'guild1', 'text', int(time.time()),
            )
            r1 = await handle_ideabox_submission(sub)
            r2 = await handle_ideabox_submission(sub)
            self.assertTrue(r2['is_duplicate'])
            self.assertEqual(r2['existing_triage_id'], r1['triage_summary'].triage_id)
        asyncio.run(_test())

    def test_invalid_input_returns_errors(self):
        async def _test():
            sub = SourceSubmission(
                '', 'user1', 'ch1', 'msg1', 'guild1', 'text', int(time.time()),
            )
            result = await handle_ideabox_submission(sub)
            self.assertFalse(result['success'])
            self.assertIn('errors', result)
        asyncio.run(_test())

    def test_malicious_instruction_canary(self):
        """Canary: external content is never treated as trusted instructions."""
        async def _test():
            sub = SourceSubmission(
                'https://github.com/owner/repo; echo "injected"',
                'user1', 'ch1', 'msg1', 'guild1', 'text', int(time.time()),
            )
            result = await handle_ideabox_submission(sub)
            # Should process as a valid GitHub URL, not execute the injection
            self.assertTrue(result['success'])
            self.assertEqual(
                result['triage_summary'].source.source_type, SourceType.GITHUB_REPO,
            )
            # The raw text is stored but never executed
            self.assertIn('echo', result['triage_summary'].source.raw_text)
        asyncio.run(_test())


class TestIdeaBoxApprovalView(unittest.TestCase):
    """Test the IdeaBoxApprovalView component generation."""

    def test_get_components(self):
        view = IdeaBoxApprovalView('t_123', {1, 2, 3})
        components = view.get_components()
        self.assertEqual(len(components), 1)  # One action row
        self.assertEqual(len(components[0]['components']), 3)  # Three buttons
        labels = [c['label'] for c in components[0]['components']]
        self.assertIn('Approve', labels)
        self.assertIn('Amend', labels)
        self.assertIn('Reject', labels)

    def test_custom_ids(self):
        view = IdeaBoxApprovalView('t_123', {1, 2, 3})
        components = view.get_components()
        for comp in components[0]['components']:
            self.assertTrue(comp['custom_id'].startswith('ideabox_approval:'))
            self.assertIn('t_123', comp['custom_id'])

    def test_disabled_components(self):
        view = IdeaBoxApprovalView('t_123', {1, 2, 3})
        disabled = view.get_disabled_components()
        for row in disabled:
            for comp in row['components']:
                self.assertTrue(comp.get('disabled', False))


if __name__ == '__main__':
    unittest.main(verbosity=2)
