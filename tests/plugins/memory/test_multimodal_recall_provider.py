import json
import threading

from plugins.memory import load_memory_provider


def test_multimodal_provider_loads():
    provider = load_memory_provider("multimodal-recall")
    assert provider is not None
    assert provider.name == "multimodal-recall"


def test_multimodal_provider_is_unavailable_without_local_mmrag(monkeypatch):
    provider = load_memory_provider("multimodal-recall")
    assert provider is not None
    provider_module = __import__(provider.__class__.__module__, fromlist=['dummy'])
    monkeypatch.setattr(provider_module, '_local_mmrag_provider_ready', lambda: False)
    assert provider.is_available() is False


def test_multimodal_provider_is_available_with_local_mmrag(monkeypatch):
    provider = load_memory_provider("multimodal-recall")
    assert provider is not None
    provider_module = __import__(provider.__class__.__module__, fromlist=['dummy'])
    monkeypatch.setattr(provider_module, '_local_mmrag_provider_ready', lambda: True)
    assert provider.is_available() is True


def test_multimodal_provider_exposes_no_tools_initially():
    provider = load_memory_provider("multimodal-recall")
    assert provider is not None
    assert provider.get_tool_schemas() == []


def test_multimodal_provider_prefetch_uses_hybrid_recall(monkeypatch, tmp_path):
    provider = load_memory_provider("multimodal-recall")
    assert provider is not None

    import sys
    provider_module = sys.modules[provider.__class__.__module__]

    monkeypatch.setattr(provider_module, '_local_mmrag_provider_ready', lambda: True)
    monkeypatch.setattr(
        provider_module,
        "recall_with_artifacts",
        lambda **kwargs: json.dumps({
            "combined_summary": "Recovered screenshot evidence for customer waiting ETA.",
            "artifact_recall": {
                "top_evidence": [
                    {"source_path": "/tmp/status.png", "source_type": "image"}
                ]
            },
        }),
    )

    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    result = provider.prefetch("that screenshot from last time")
    assert "Recovered screenshot evidence" in result
    assert "status.png" in result
    assert len(result) < 800


def test_multimodal_provider_prefetch_includes_issue_source_ref_when_available(monkeypatch, tmp_path):
    provider = load_memory_provider("multimodal-recall")
    assert provider is not None

    import sys
    provider_module = sys.modules[provider.__class__.__module__]

    monkeypatch.setattr(provider_module, '_local_mmrag_provider_ready', lambda: True)
    monkeypatch.setattr(
        provider_module,
        "recall_with_artifacts",
        lambda **kwargs: json.dumps({
            "combined_summary": "Recovered screenshot evidence for customer waiting ETA.",
            "artifact_recall": {
                "top_evidence": [
                    {
                        "source_path": "/tmp/status.png",
                        "source_type": "image",
                        "source_ref": "issue:987",
                        "collection_name": "gitlab_radar_demo",
                    }
                ]
            },
        }),
    )

    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    result = provider.prefetch("that screenshot from last time")
    assert "issue:987" in result
    assert "gitlab_radar_demo" in result


def test_multimodal_provider_prefetch_skips_non_multimodal_queries(tmp_path):
    provider = load_memory_provider("multimodal-recall")
    assert provider is not None
    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    assert provider.prefetch("what branch name did we use") == ""


def test_multimodal_provider_prefetch_skips_weak_artifact_queries_without_recent_signal(monkeypatch, tmp_path):
    provider = load_memory_provider("multimodal-recall")
    assert provider is not None
    provider_module = __import__(provider.__class__.__module__, fromlist=['dummy'])

    monkeypatch.setattr(provider_module, '_local_mmrag_provider_ready', lambda: True)
    called = {'value': False}

    def _fake_recall(**kwargs):
        called['value'] = True
        return json.dumps({'combined_summary': 'should not be used', 'artifact_recall': {'top_evidence': []}})

    monkeypatch.setattr(provider_module, 'recall_with_artifacts', _fake_recall)

    provider.initialize("session-1", hermes_home=str(tmp_path), platform="cli")
    assert provider.prefetch("what was the ETA from that file") == ""
    assert called['value'] is False


def test_multimodal_provider_prefetch_uses_recent_signal_boost_for_weak_artifact_queries(monkeypatch, tmp_path):
    provider = load_memory_provider("multimodal-recall")
    assert provider is not None

    import sys
    provider_module = sys.modules[provider.__class__.__module__]

    monkeypatch.setattr(provider_module, '_local_mmrag_provider_ready', lambda: True)
    monkeypatch.setattr(
        provider_module,
        'recall_with_artifacts',
        lambda **kwargs: json.dumps({
            'combined_summary': 'Recovered file ETA from prior screenshot context.',
            'artifact_recall': {'top_evidence': [{'source_path': '/tmp/report.png'}]},
        }),
    )

    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')
    provider.sync_turn(
        'please remember that screenshot evidence from the PDF review',
        'I checked the attachment and OCR evidence.',
        session_id='session-1',
    )
    result = provider.prefetch('what was the ETA from that file')
    assert 'Recovered file ETA' in result
    assert 'report.png' in result


def test_memory_manager_prefetch_all_uses_multimodal_provider_after_runtime_style_init(monkeypatch, tmp_path):
    from agent.memory_manager import MemoryManager

    provider = load_memory_provider("multimodal-recall")
    assert provider is not None
    provider_module = __import__(provider.__class__.__module__, fromlist=['dummy'])

    monkeypatch.setattr(provider_module, '_local_mmrag_provider_ready', lambda: True)
    monkeypatch.setattr(
        provider_module,
        'recall_with_artifacts',
        lambda **kwargs: json.dumps({
            'combined_summary': 'Recovered attachment evidence from prior work.',
            'artifact_recall': {'top_evidence': [{'source_path': '/tmp/evidence.png'}]},
        }),
    )

    mgr = MemoryManager()
    mgr.add_provider(provider)
    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')

    result = mgr.prefetch_all('that attachment from last time', session_id='session-1')
    assert 'Recovered attachment evidence' in result
    assert 'evidence.png' in result


def test_multimodal_provider_prefetch_returns_empty_if_runtime_connectivity_drops(monkeypatch, tmp_path):
    provider = load_memory_provider('multimodal-recall')
    assert provider is not None
    provider_module = __import__(provider.__class__.__module__, fromlist=['dummy'])

    monkeypatch.setattr(provider_module, '_local_mmrag_provider_ready', lambda: False)
    called = {'value': False}

    def _fake_recall(**kwargs):
        called['value'] = True
        return json.dumps({'combined_summary': 'should not be used'})

    monkeypatch.setattr(provider_module, 'recall_with_artifacts', _fake_recall)

    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')
    result = provider.prefetch('that screenshot from last time')
    assert result == ''
    assert called['value'] is False


def test_multimodal_provider_queue_prefetch_caches_next_turn_context(monkeypatch, tmp_path):
    provider = load_memory_provider('multimodal-recall')
    assert provider is not None
    provider_module = __import__(provider.__class__.__module__, fromlist=['dummy'])

    monkeypatch.setattr(provider_module, '_local_mmrag_provider_ready', lambda: True)
    calls = {'count': 0}

    def _fake_recall(**kwargs):
        calls['count'] += 1
        return json.dumps({
            'combined_summary': 'Prefetched screenshot evidence.',
            'artifact_recall': {'top_evidence': [{'source_path': '/tmp/prefetch.png'}]},
        })

    monkeypatch.setattr(provider_module, 'recall_with_artifacts', _fake_recall)

    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')
    provider.queue_prefetch('that screenshot from last time')
    provider._prefetch_thread.join(timeout=1)
    assert calls['count'] == 1
    result = provider.prefetch('that screenshot from last time')
    assert 'Prefetched screenshot evidence.' in result
    assert 'prefetch.png' in result
    assert calls['count'] == 1


def test_multimodal_provider_prefetch_cache_key_normalizes_equivalent_queries(monkeypatch, tmp_path):
    provider = load_memory_provider('multimodal-recall')
    assert provider is not None
    provider_module = __import__(provider.__class__.__module__, fromlist=['dummy'])

    monkeypatch.setattr(provider_module, '_local_mmrag_provider_ready', lambda: True)
    calls = {'count': 0}

    def _fake_recall(**kwargs):
        calls['count'] += 1
        return json.dumps({
            'combined_summary': 'Prefetched normalized screenshot evidence.',
            'artifact_recall': {'top_evidence': [{'source_path': '/tmp/normalized.png'}]},
        })

    monkeypatch.setattr(provider_module, 'recall_with_artifacts', _fake_recall)

    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')
    provider.queue_prefetch('  That Screenshot   From Last Time  ')
    provider._prefetch_thread.join(timeout=1)

    result = provider.prefetch('that screenshot from last time')

    assert 'Prefetched normalized screenshot evidence.' in result
    assert 'normalized.png' in result
    assert calls['count'] == 1


def test_multimodal_provider_queue_prefetch_is_non_blocking(monkeypatch, tmp_path):
    provider = load_memory_provider('multimodal-recall')
    assert provider is not None
    provider_module = __import__(provider.__class__.__module__, fromlist=['dummy'])

    monkeypatch.setattr(provider_module, '_local_mmrag_provider_ready', lambda: True)
    started = threading.Event()
    release = threading.Event()

    def _slow_recall(**kwargs):
        started.set()
        release.wait(timeout=1)
        return json.dumps({'combined_summary': 'slow prefetch', 'artifact_recall': {'top_evidence': []}})

    monkeypatch.setattr(provider_module, 'recall_with_artifacts', _slow_recall)

    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')
    provider.queue_prefetch('that screenshot from last time')
    assert provider._prefetch_thread is not None
    assert started.wait(timeout=1)
    # queue_prefetch should have returned already; thread still running until released
    assert provider._prefetch_thread.is_alive()
    release.set()
    provider._prefetch_thread.join(timeout=1)


def test_multimodal_provider_queue_prefetch_swallows_background_recall_errors(monkeypatch, tmp_path, recwarn):
    provider = load_memory_provider('multimodal-recall')
    assert provider is not None
    provider_module = __import__(provider.__class__.__module__, fromlist=['dummy'])

    monkeypatch.setattr(provider_module, '_local_mmrag_provider_ready', lambda: True)

    def _boom(**kwargs):
        raise RuntimeError('background recall failed')

    monkeypatch.setattr(provider_module, 'recall_with_artifacts', _boom)

    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')
    provider.queue_prefetch('that screenshot from last time')
    provider._prefetch_thread.join(timeout=1)

    assert not recwarn.list
    assert provider.prefetch('that screenshot from last time') == ''


def test_multimodal_provider_prefetch_consumes_cache_once_then_recalls_again(monkeypatch, tmp_path):
    provider = load_memory_provider('multimodal-recall')
    assert provider is not None
    provider_module = __import__(provider.__class__.__module__, fromlist=['dummy'])

    monkeypatch.setattr(provider_module, '_local_mmrag_provider_ready', lambda: True)
    calls = {'count': 0}

    def _fake_recall(**kwargs):
        calls['count'] += 1
        return json.dumps({
            'combined_summary': f'Prefetch call {calls["count"]}',
            'artifact_recall': {'top_evidence': []},
        })

    monkeypatch.setattr(provider_module, 'recall_with_artifacts', _fake_recall)

    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')
    provider._prefetch_cooldown_seconds = 0.0
    provider.queue_prefetch('that screenshot from last time')
    first = provider.prefetch('that screenshot from last time')
    second = provider.prefetch('that screenshot from last time')

    assert 'Prefetch call 1' in first
    assert 'Prefetch call 2' in second
    assert calls['count'] == 2


def test_multimodal_provider_prefetch_applies_cooldown_to_repeated_fresh_recalls(monkeypatch, tmp_path):
    provider = load_memory_provider('multimodal-recall')
    assert provider is not None
    provider_module = __import__(provider.__class__.__module__, fromlist=['dummy'])

    monkeypatch.setattr(provider_module, '_local_mmrag_provider_ready', lambda: True)
    calls = {'count': 0}

    def _fake_recall(**kwargs):
        calls['count'] += 1
        return json.dumps({
            'combined_summary': f'Fresh recall {calls["count"]}',
            'artifact_recall': {'top_evidence': []},
        })

    monkeypatch.setattr(provider_module, 'recall_with_artifacts', _fake_recall)

    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')
    first = provider.prefetch('that screenshot from last time')
    second = provider.prefetch('that screenshot from last time')

    assert 'Fresh recall 1' in first
    assert second == ''
    assert calls['count'] == 1


def test_multimodal_provider_on_session_end_captures_lightweight_signal(tmp_path):
    provider = load_memory_provider('multimodal-recall')
    assert provider is not None
    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')

    messages = [
        {'role': 'user', 'content': 'please remember that screenshot evidence from the PDF review'},
        {'role': 'assistant', 'content': 'I checked the attachment and OCR evidence.'},
    ]
    provider.on_session_end(messages)

    assert provider._session_end_signal
    assert 'screenshot' in provider._session_end_signal.lower() or 'evidence' in provider._session_end_signal.lower()
    assert len(provider._session_end_signal) < 400


def test_multimodal_provider_on_session_end_ignores_non_multimodal_sessions(tmp_path):
    provider = load_memory_provider('multimodal-recall')
    assert provider is not None
    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')

    messages = [
        {'role': 'user', 'content': 'what branch name did we use'},
        {'role': 'assistant', 'content': 'we used feature/test-branch'},
    ]
    provider.on_session_end(messages)

    assert provider._session_end_signal == ''


def test_multimodal_provider_session_end_signal_boosts_weak_artifact_queries(monkeypatch, tmp_path):
    provider = load_memory_provider('multimodal-recall')
    assert provider is not None
    provider_module = __import__(provider.__class__.__module__, fromlist=['dummy'])

    monkeypatch.setattr(provider_module, '_local_mmrag_provider_ready', lambda: True)
    monkeypatch.setattr(
        provider_module,
        'recall_with_artifacts',
        lambda **kwargs: json.dumps({
            'combined_summary': 'Recovered file ETA from prior session-end screenshot context.',
            'artifact_recall': {'top_evidence': [{'source_path': '/tmp/session-end.png'}]},
        }),
    )

    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')
    provider.on_session_end([
        {'role': 'user', 'content': 'please remember that screenshot evidence from the PDF review'},
        {'role': 'assistant', 'content': 'I checked the attachment and OCR evidence.'},
    ])
    result = provider.prefetch('what was the ETA from that file')
    assert 'Recovered file ETA' in result
    assert 'session-end.png' in result


def test_multimodal_provider_sync_turn_captures_lightweight_signal(tmp_path):
    provider = load_memory_provider('multimodal-recall')
    assert provider is not None
    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')

    provider.sync_turn(
        'please remember that screenshot evidence from the PDF review',
        'I checked the attachment and OCR evidence.',
        session_id='session-1',
    )

    assert provider._turn_signal
    assert 'screenshot' in provider._turn_signal.lower() or 'evidence' in provider._turn_signal.lower()
    assert len(provider._turn_signal) < 400


def test_multimodal_provider_sync_turn_ignores_non_multimodal_turns(tmp_path):
    provider = load_memory_provider('multimodal-recall')
    assert provider is not None
    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')

    provider.sync_turn('what branch name did we use', 'we used feature/test-branch', session_id='session-1')
    assert provider._turn_signal == ''


def test_multimodal_provider_on_memory_write_captures_lightweight_signal(tmp_path):
    provider = load_memory_provider('multimodal-recall')
    assert provider is not None
    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')

    provider.on_memory_write('add', 'memory', 'Remember the screenshot evidence from the PDF review.')
    assert provider._memory_write_signal
    assert 'screenshot' in provider._memory_write_signal.lower() or 'evidence' in provider._memory_write_signal.lower()
    assert len(provider._memory_write_signal) < 400


def test_multimodal_provider_on_memory_write_ignores_non_multimodal_content(tmp_path):
    provider = load_memory_provider('multimodal-recall')
    assert provider is not None
    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')

    provider.on_memory_write('add', 'memory', 'Remember that the user prefers concise answers.')
    assert provider._memory_write_signal == ''


def test_multimodal_provider_memory_write_signal_boosts_weak_artifact_queries(monkeypatch, tmp_path):
    provider = load_memory_provider('multimodal-recall')
    assert provider is not None
    provider_module = __import__(provider.__class__.__module__, fromlist=['dummy'])

    monkeypatch.setattr(provider_module, '_local_mmrag_provider_ready', lambda: True)
    monkeypatch.setattr(
        provider_module,
        'recall_with_artifacts',
        lambda **kwargs: json.dumps({
            'combined_summary': 'Recovered file ETA from prior memory-write screenshot context.',
            'artifact_recall': {'top_evidence': [{'source_path': '/tmp/memory-write.png'}]},
        }),
    )

    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')
    provider.on_memory_write('add', 'memory', 'Remember the screenshot evidence from the PDF review.')
    result = provider.prefetch('what was the ETA from that file')
    assert 'Recovered file ETA' in result
    assert 'memory-write.png' in result


def test_multimodal_provider_shutdown_joins_and_clears_prefetch_thread(monkeypatch, tmp_path):
    provider = load_memory_provider('multimodal-recall')
    assert provider is not None
    provider_module = __import__(provider.__class__.__module__, fromlist=['dummy'])

    monkeypatch.setattr(provider_module, '_local_mmrag_provider_ready', lambda: True)
    started = threading.Event()
    release = threading.Event()

    def _slow_recall(**kwargs):
        started.set()
        release.wait(timeout=1)
        return json.dumps({'combined_summary': 'slow prefetch', 'artifact_recall': {'top_evidence': []}})

    monkeypatch.setattr(provider_module, 'recall_with_artifacts', _slow_recall)

    provider.initialize('session-1', hermes_home=str(tmp_path), platform='cli')
    provider.queue_prefetch('that screenshot from last time')
    assert started.wait(timeout=1)
    assert provider._prefetch_thread is not None

    release.set()
    provider.shutdown()
    assert provider._prefetch_thread is None
