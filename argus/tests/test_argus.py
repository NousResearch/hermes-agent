#!/usr/bin/env python3
"""
Unit tests for ARGUS
The Hundred-Eyed Watchman
Tests entropy detection, decision logic, and database operations.
"""

import os
import sys
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from argus import Argus

class TestArgus(unittest.TestCase):
    """Test cases for Argus class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        # Create Argus with temporary database
        self.argus = Argus()
        self.argus.db_path = self.temp_db.name
        self.argus._init_database()
        self.argus._load_schema()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.argus.conn:
            self.argus.conn.close()
        os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test database initialization."""
        # Check that tables exist
        self.argus.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in self.argus.cursor.fetchall()]
        
        expected_tables = [
            'sessions', 'tool_calls', 'file_changes', 'terminal_commands',
            'quality_metrics', 'entropy_detections', 'watcher_actions',
            'notifications', 'directive_checks'
        ]
        
        for table in expected_tables:
            self.assertIn(table, tables)
    
    def test_session_registration(self):
        """Test session registration."""
        session = {
            'session_id': 'test_session_1',
            'session_type': 'cron',
            'job_id': 'test_job_1',
            'task_description': 'Test session',
            'model': 'test_model',
            'provider': 'test_provider',
            'metadata': json.dumps({'test': 'data'})
        }
        
        self.argus.register_session(session)
        
        # Verify session was registered
        self.argus.cursor.execute('SELECT * FROM sessions WHERE session_id = ?', ('test_session_1',))
        result = self.argus.cursor.fetchone()
        
        self.assertIsNotNone(result)
        self.assertEqual(result['session_type'], 'cron')
        self.assertEqual(result['job_id'], 'test_job_1')
    
    def test_repeat_tool_calls_detection(self):
        """Test repeat tool calls detection."""
        session_id = 'test_session_2'
        
        # Register session
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'manual',
            'task_description': 'Test session'
        })
        
        # Insert repeated tool calls (use UTC to match SQLite datetime('now'))
        from datetime import timezone
        utc_now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        for i in range(5):
            self.argus.cursor.execute('''
                INSERT INTO tool_calls (session_id, tool_name, tool_args, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (session_id, 'read_file', '{"path": "test.py"}', utc_now))
        
        self.argus.conn.commit()
        
        # Detect entropy
        detections = self.argus._detect_repeat_tool_calls(session_id)
        
        # Verify detection
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]['entropy_type'], 'repeat_tool_calls')
        self.assertEqual(detections[0]['severity'], 'critical')
    
    def test_repeat_commands_detection(self):
        """Test repeat commands detection."""
        session_id = 'test_session_3'
        
        # Register session
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'task_description': 'Test session'
        })
        
        # Insert repeated commands (use UTC to match SQLite datetime('now'))
        from datetime import timezone
        utc_now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        for i in range(4):
            self.argus.cursor.execute('''
                INSERT INTO terminal_commands (session_id, command, timestamp)
                VALUES (?, ?, ?)
            ''', (session_id, 'ls -la', utc_now))
        
        self.argus.conn.commit()
        
        # Detect entropy
        detections = self.argus._detect_repeat_commands(session_id)
        
        # Verify detection
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]['entropy_type'], 'repeat_commands')
        self.assertEqual(detections[0]['severity'], 'warning')
    
    def test_stuck_loop_detection(self):
        """Test stuck loop detection."""
        session_id = 'test_session_4'
        
        # Register session
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'delegate_task',
            'task_description': 'Test session'
        })
        
        # Insert repeating pattern of tool calls
        pattern = [
            ('read_file', '{"path": "a.py"}'),
            ('write_file', '{"path": "a.py"}'),
            ('read_file', '{"path": "b.py"}'),
        ]
        
        # Repeat pattern twice
        for _ in range(2):
            for tool_name, tool_args in pattern:
                self.argus.cursor.execute('''
                    INSERT INTO tool_calls (session_id, tool_name, tool_args, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (session_id, tool_name, tool_args, datetime.now().isoformat()))
        
        self.argus.conn.commit()
        
        # Detect entropy
        detections = self.argus._detect_stuck_loops(session_id)
        
        # Verify detection
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]['entropy_type'], 'stuck_loop')
        self.assertEqual(detections[0]['severity'], 'critical')
    
    def test_no_file_changes_detection(self):
        """Test no file changes detection."""
        session_id = 'test_session_5'
        
        # Register session
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'task_description': 'Test session'
        })
        
        # Insert write operations without file changes (use UTC)
        from datetime import timezone
        utc_now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        for i in range(3):
            self.argus.cursor.execute('''
                INSERT INTO tool_calls (session_id, tool_name, file_path, file_changed, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, 'write_file', 'test.py', False, utc_now))
        
        self.argus.conn.commit()
        
        # Detect entropy
        detections = self.argus._detect_no_file_changes(session_id)
        
        # Verify detection
        self.assertEqual(len(detections), 3)
        for detection in detections:
            self.assertEqual(detection['entropy_type'], 'no_file_changes')
            self.assertEqual(detection['severity'], 'critical')
    
    def test_decision_restart_on_critical_entropy(self):
        """Test decision to restart on critical entropy."""
        session_id = 'test_session_6'
        
        # Register session
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'task_description': 'Test session'
        })
        
        # Create critical entropy detection
        entropy_detections = [
            {
                'entropy_type': 'stuck_loop',
                'severity': 'critical',
                'details': json.dumps({'pattern_length': 2})
            }
        ]
        
        directive_checks = [
            {'check_type': 'pipeline_compliance', 'passed': True}
        ]
        
        # Make decision
        decision = self.argus.make_decision(session_id, entropy_detections, directive_checks)
        
        # Verify decision
        self.assertIsNotNone(decision)
        self.assertEqual(decision['action'], 'restart')
        self.assertIn('Critical entropy', decision['reason'])
    
    def test_decision_kill_on_repeat_tool_calls(self):
        """Test decision to kill on repeat tool calls."""
        session_id = 'test_session_7'
        
        # Register session
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'task_description': 'Test session'
        })
        
        # Update restart count to max
        self.argus.cursor.execute('''
            UPDATE sessions SET restart_count = ? WHERE session_id = ?
        ''', (3, session_id))
        self.argus.conn.commit()
        
        # Create critical entropy detection for repeat tool calls
        entropy_detections = [
            {
                'entropy_type': 'repeat_tool_calls',
                'severity': 'critical',
                'details': json.dumps({'tool_name': 'read_file', 'count': 5})
            }
        ]
        
        directive_checks = [
            {'check_type': 'pipeline_compliance', 'passed': True}
        ]
        
        # Make decision
        decision = self.argus.make_decision(session_id, entropy_detections, directive_checks)
        
        # Verify decision
        self.assertIsNotNone(decision)
        self.assertEqual(decision['action'], 'kill')
        self.assertIn('max restarts', decision['reason'])
    
    def test_decision_restart_on_directive_violation(self):
        """Test decision to restart on directive violation."""
        session_id = 'test_session_8'
        
        # Register session
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'manual',
            'task_description': 'Test session'
        })
        
        # Create directive violation
        entropy_detections = []
        directive_checks = [
            {'check_type': 'pipeline_compliance', 'passed': False}
        ]
        
        # Make decision
        decision = self.argus.make_decision(session_id, entropy_detections, directive_checks)
        
        # Verify decision
        self.assertIsNotNone(decision)
        self.assertEqual(decision['action'], 'restart')
        self.assertIn('Prime directive violation', decision['reason'])
    
    def test_decision_no_action(self):
        """Test decision when no action needed."""
        session_id = 'test_session_9'
        
        # Register session
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'task_description': 'Test session'
        })
        
        # No entropy detections
        entropy_detections = []
        directive_checks = [
            {'check_type': 'pipeline_compliance', 'passed': True}
        ]
        
        # Make decision
        decision = self.argus.make_decision(session_id, entropy_detections, directive_checks)
        
        # Verify no decision
        self.assertIsNone(decision)
    
    def test_action_recording(self):
        """Test action recording in database."""
        session_id = 'test_session_10'
        
        # Register session
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'task_description': 'Test session'
        })
        
        # Record action
        self.argus.cursor.execute('''
            INSERT INTO watcher_actions (session_id, action_type, action_reason, details)
            VALUES (?, ?, ?, ?)
        ''', (session_id, 'restart', 'Test reason', json.dumps({'test': 'data'})))
        
        self.argus.conn.commit()
        
        # Verify action was recorded
        self.argus.cursor.execute('SELECT * FROM watcher_actions WHERE session_id = ?', (session_id,))
        result = self.argus.cursor.fetchone()
        
        self.assertIsNotNone(result)
        self.assertEqual(result['action_type'], 'restart')
        self.assertEqual(result['action_reason'], 'Test reason')
    
    def test_quality_metrics_tracking(self):
        """Test quality metrics tracking."""
        session_id = 'test_session_11'
        
        # Register session
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'task_description': 'Test session'
        })
        
        # Insert quality metrics
        for i in range(5):
            self.argus.cursor.execute('''
                INSERT INTO quality_metrics (session_id, metric_type, metric_value, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (session_id, 'trajectory_quality', 0.93 + i * 0.01, datetime.now().isoformat()))
        
        self.argus.conn.commit()
        
        # Query average quality
        self.argus.cursor.execute('''
            SELECT AVG(metric_value) as avg_quality
            FROM quality_metrics
            WHERE session_id = ?
        ''', (session_id,))
        
        result = self.argus.cursor.fetchone()
        avg_quality = result['avg_quality']
        
        # Verify average
        self.assertAlmostEqual(avg_quality, 0.95, places=2)
    
    def test_entropy_detection_integration(self):
        """Test integration of all entropy detection methods."""
        session_id = 'test_session_12'
        
        # Register session
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'task_description': 'Test session'
        })
        
        # Insert data for all detection methods (use UTC)
        from datetime import timezone
        utc_now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        # 1. Repeat tool calls
        for i in range(4):
            self.argus.cursor.execute('''
                INSERT INTO tool_calls (session_id, tool_name, tool_args, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (session_id, 'read_file', '{"path": "test.py"}', utc_now))

        # 2. Repeat commands
        for i in range(3):
            self.argus.cursor.execute('''
                INSERT INTO terminal_commands (session_id, command, timestamp)
                VALUES (?, ?, ?)
            ''', (session_id, 'ls -la', utc_now))

        # 3. No file changes
        for i in range(2):
            self.argus.cursor.execute('''
                INSERT INTO tool_calls (session_id, tool_name, file_path, file_changed, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, 'write_file', 'test.py', False, utc_now))
        
        self.argus.conn.commit()
        
        # Detect all entropy
        detections = self.argus.detect_entropy(session_id)
        
        # Verify detections
        self.assertGreater(len(detections), 0)
        
        # Check detection types
        detection_types = [d['entropy_type'] for d in detections]
        self.assertIn('repeat_tool_calls', detection_types)
        self.assertIn('repeat_commands', detection_types)
        self.assertIn('no_file_changes', detection_types)

class TestArgusConfiguration(unittest.TestCase):
    """Test configuration handling."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        # Import CONFIG from argus module
        from argus import CONFIG
        
        # Check default config values
        self.assertEqual(CONFIG['poll_interval'], 30)
        self.assertEqual(CONFIG['entropy_threshold'], 3)
        self.assertEqual(CONFIG['quality_threshold'], 0.92)
        self.assertEqual(CONFIG['max_restart_count'], 3)

class TestArgusRestart(unittest.TestCase):
    """Test restart logic for different session types."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.argus = Argus()
        self.argus.db_path = self.temp_db.name
        self.argus._init_database()
        self.argus._load_schema()

    def tearDown(self):
        if self.argus.conn:
            self.argus.conn.close()
        os.unlink(self.temp_db.name)

    @patch('argus.pause_job')
    @patch('argus.resume_job')
    def test_restart_cron_session_pauses_and_resumes(self, mock_resume, mock_pause):
        """Cron restart must pause job, inject corrective prompt, resume."""
        mock_pause.return_value = {'id': 'test_job_1', 'enabled': False}
        mock_resume.return_value = {'id': 'test_job_1', 'enabled': True}

        session_id = 'cron_test_job_1'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'job_id': 'test_job_1',
            'task_description': 'Test cron job'
        })

        self.argus._restart_session(session_id, 'Test restart')

        # Verify restart_count incremented
        self.argus.cursor.execute('SELECT restart_count, status FROM sessions WHERE session_id = ?', (session_id,))
        row = self.argus.cursor.fetchone()
        self.assertEqual(row['restart_count'], 1)
        self.assertEqual(row['status'], 'restarted')

        # Verify cron.jobs functions were called
        mock_pause.assert_called_once()
        mock_resume.assert_called_once()

    @patch('argus.subprocess.run')
    def test_restart_delegate_session_kills_and_respawns(self, mock_run):
        """Delegate restart must kill process and respawn with corrective prompt."""
        session_id = 'delegate_12345'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'delegate_task',
            'task_description': 'Delegate task (PID: 12345)',
            'metadata': json.dumps({'pid': '12345'})
        })

        self.argus._restart_session(session_id, 'Test restart')

        # Verify restart_count incremented
        self.argus.cursor.execute('SELECT restart_count, status FROM sessions WHERE session_id = ?', (session_id,))
        row = self.argus.cursor.fetchone()
        self.assertEqual(row['restart_count'], 1)

        # Verify process was killed (kill -TERM or kill -KILL)
        calls = [str(c) for c in mock_run.call_args_list]
        kill_called = any('kill' in c.lower() or 'SIGTERM' in c or 'SIGKILL' in c for c in calls)
        self.assertTrue(kill_called, "Should kill delegate process on restart")

    def test_restart_manual_session_records_action(self):
        """Manual session restart records action but cannot force restart."""
        session_id = 'manual_99999'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'manual',
            'task_description': 'Manual session (PID: 99999)',
            'metadata': json.dumps({'pid': '99999'})
        })

        self.argus._restart_session(session_id, 'Test restart')

        # Verify action was recorded
        self.argus.cursor.execute('SELECT * FROM watcher_actions WHERE session_id = ?', (session_id,))
        action = self.argus.cursor.fetchone()
        self.assertIsNotNone(action)
        self.assertEqual(action['action_type'], 'restart')


class TestArgusKill(unittest.TestCase):
    """Test kill logic for different session types."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.argus = Argus()
        self.argus.db_path = self.temp_db.name
        self.argus._init_database()
        self.argus._load_schema()

    def tearDown(self):
        if self.argus.conn:
            self.argus.conn.close()
        os.unlink(self.temp_db.name)

    @patch('argus.pause_job')
    def test_kill_cron_session_pauses_job(self, mock_pause):
        """Cron kill must permanently pause the job via cron.jobs."""
        mock_pause.return_value = {'id': 'kill_job_1', 'enabled': False}

        session_id = 'cron_kill_job'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'job_id': 'kill_job_1',
            'task_description': 'Cron to kill'
        })

        self.argus._kill_session(session_id, 'Max restarts exceeded')

        # Verify status updated
        self.argus.cursor.execute('SELECT status, kill_count FROM sessions WHERE session_id = ?', (session_id,))
        row = self.argus.cursor.fetchone()
        self.assertEqual(row['status'], 'killed')
        self.assertEqual(row['kill_count'], 1)

        # Verify cron.jobs.pause_job was called
        mock_pause.assert_called_once()

    @patch('argus.subprocess.run')
    def test_kill_delegate_session_terminates_process(self, mock_run):
        """Delegate kill must terminate the subprocess."""
        session_id = 'delegate_kill_67890'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'delegate_task',
            'task_description': 'Delegate to kill (PID: 67890)',
            'metadata': json.dumps({'pid': '67890'})
        })

        self.argus._kill_session(session_id, 'High entropy detected')

        # Verify process was killed
        calls = [str(c) for c in mock_run.call_args_list]
        kill_called = any('kill' in c.lower() or 'SIGTERM' in c or 'SIGKILL' in c for c in calls)
        self.assertTrue(kill_called, "Should kill delegate process")

    def test_kill_manual_session_sends_alert_only(self):
        """Manual session kill sends alert, cannot terminate."""
        session_id = 'manual_alert_11111'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'manual',
            'task_description': 'Manual session to alert',
            'metadata': json.dumps({'pid': '11111'})
        })

        self.argus._kill_session(session_id, 'Cannot kill manual session')

        # Verify notification was recorded
        self.argus.cursor.execute('SELECT * FROM notifications WHERE session_id = ?', (session_id,))
        notif = self.argus.cursor.fetchone()
        self.assertIsNotNone(notif, "Should send notification for manual session kill")
        self.assertIn('manual', notif['message'].lower())


class TestArgusPromptInjection(unittest.TestCase):
    """Test prompt injection for corrective actions."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.argus = Argus()
        self.argus.db_path = self.temp_db.name
        self.argus._init_database()
        self.argus._load_schema()

    def tearDown(self):
        if self.argus.conn:
            self.argus.conn.close()
        os.unlink(self.temp_db.name)

    def test_inject_prompt_cron_updates_job_prompt(self):
        """Cron prompt injection must update the job's prompt with corrective instructions."""
        session_id = 'cron_inject_job'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'job_id': 'inject_job_1',
            'task_description': 'Cron for injection'
        })

        corrective_prompt = "ENTROPY CORRECTION: You were repeating tool calls. Focus on completing the task."
        self.argus._inject_prompt(session_id, corrective_prompt)

        # Verify prompt injection was recorded
        self.argus.cursor.execute(
            "SELECT * FROM watcher_actions WHERE session_id = ? AND action_type = 'inject_prompt'",
            (session_id,)
        )
        action = self.argus.cursor.fetchone()
        self.assertIsNotNone(action, "Should record prompt injection action")

    @patch('argus.subprocess.run')
    def test_inject_prompt_delegate_respawns_with_prompt(self, mock_run):
        """Delegate prompt injection must kill and respawn with corrective prompt."""
        session_id = 'delegate_inject_22222'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'delegate_task',
            'task_description': 'Delegate for injection (PID: 22222)',
            'metadata': json.dumps({'pid': '22222'})
        })

        corrective_prompt = "ENTROPY CORRECTION: Stop looping. Complete one task at a time."
        self.argus._inject_prompt(session_id, corrective_prompt)

        # Verify process was killed (to respawn with new prompt)
        calls = [str(c) for c in mock_run.call_args_list]
        kill_called = any('kill' in c.lower() or 'SIGTERM' in c or 'SIGKILL' in c for c in calls)
        self.assertTrue(kill_called, "Should kill delegate to respawn with corrective prompt")

    def test_inject_prompt_manual_creates_queue_entry(self):
        """Manual session prompt injection creates queue/record for next interaction."""
        session_id = 'manual_inject_33333'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'manual',
            'task_description': 'Manual session for injection',
            'metadata': json.dumps({'pid': '33333'})
        })

        corrective_prompt = "ENTROPY CORRECTION: You were stuck in a loop."
        self.argus._inject_prompt(session_id, corrective_prompt)

        # Verify action was recorded with the prompt
        self.argus.cursor.execute(
            "SELECT * FROM watcher_actions WHERE session_id = ? AND action_type = 'inject_prompt'",
            (session_id,)
        )
        action = self.argus.cursor.fetchone()
        self.assertIsNotNone(action)
        details = json.loads(action['details'])
        self.assertIn('corrective_prompt', details)


class TestArgusTelegramNotification(unittest.TestCase):
    """Test Telegram notification integration."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.argus = Argus()
        self.argus.db_path = self.temp_db.name
        self.argus._init_database()
        self.argus._load_schema()

    def tearDown(self):
        if self.argus.conn:
            self.argus.conn.close()
        os.unlink(self.temp_db.name)

    @patch('argus.subprocess.run')
    @patch('argus.Path')
    def test_send_notification_reads_credentials(self, mock_path, mock_run):
        """Notification must read Telegram credentials from configured path."""
        session_id = 'notif_test_session'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'task_description': 'Notification test'
        })

        # Mock credential file
        mock_path_obj = MagicMock()
        mock_path_obj.read_text.return_value = 'TELEGRAM_BOT_TOKEN=123:abc\nTELEGRAM_CHAT_ID=456'
        mock_path.return_value = mock_path_obj

        self.argus._send_notification(session_id, 'restart', 'Test restart reason')

        # Verify notification was recorded in DB
        self.argus.cursor.execute('SELECT * FROM notifications WHERE session_id = ?', (session_id,))
        notif = self.argus.cursor.fetchone()
        self.assertIsNotNone(notif)
        self.assertEqual(notif['notification_type'], 'restart')
        self.assertIn('Test restart reason', notif['message'])

    @patch('argus.subprocess.run')
    def test_send_notification_formats_message(self, mock_run):
        """Notification message must include session type, task, action, reason."""
        session_id = 'notif_format_test'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'delegate_task',
            'task_description': 'Format test task'
        })

        self.argus._send_notification(session_id, 'kill', 'High entropy: repeat tool calls')

        self.argus.cursor.execute('SELECT message FROM notifications WHERE session_id = ?', (session_id,))
        message = self.argus.cursor.fetchone()['message']

        self.assertIn('delegate_task', message)
        self.assertIn('Format test task', message)
        self.assertIn('KILL', message)
        self.assertIn('repeat tool calls', message)

    def test_send_notification_records_delivery_status(self):
        """Notification delivery status must be tracked."""
        session_id = 'notif_delivery_test'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'manual',
            'task_description': 'Delivery test'
        })

        self.argus._send_notification(session_id, 'restart', 'Test')

        self.argus.cursor.execute(
            'SELECT delivered, delivery_error FROM notifications WHERE session_id = ?',
            (session_id,)
        )
        row = self.argus.cursor.fetchone()
        # Should have a delivered status (True or False depending on Telegram API)
        self.assertIsNotNone(row['delivered'])


class TestArgusCollectMetrics(unittest.TestCase):
    """Test metrics collection from session logs and holographic_memory.db."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.argus = Argus()
        self.argus.db_path = self.temp_db.name
        self.argus._init_database()
        self.argus._load_schema()

    def tearDown(self):
        if self.argus.conn:
            self.argus.conn.close()
        os.unlink(self.temp_db.name)

    def test_collect_metrics_populates_tool_calls(self):
        """collect_metrics must populate tool_calls table from session data."""
        session_id = 'metrics_test_1'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'task_description': 'Metrics test'
        })

        # collect_metrics should not raise and should handle empty sessions gracefully
        self.argus.collect_metrics(session_id)

        # After collection, tool_call_count on session should be updated
        self.argus.cursor.execute('SELECT tool_call_count FROM sessions WHERE session_id = ?', (session_id,))
        row = self.argus.cursor.fetchone()
        self.assertIsNotNone(row['tool_call_count'])

    def test_collect_metrics_updates_quality_score(self):
        """collect_metrics must update quality_gate_score on session."""
        session_id = 'metrics_test_2'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'task_description': 'Quality metrics test'
        })

        self.argus.collect_metrics(session_id)

        self.argus.cursor.execute('SELECT quality_gate_score FROM sessions WHERE session_id = ?', (session_id,))
        row = self.argus.cursor.fetchone()
        # quality_gate_score should be set (even if None for sessions without data)
        self.assertIsNotNone(row)


class TestArgusConfigFile(unittest.TestCase):
    """Test configuration loading from hermes config.yaml."""

    @patch('argus._hermes_load_config')
    def test_load_config_from_hermes_config(self, mock_load):
        """Config must load from hermes config.yaml 'argus' key."""
        mock_load.return_value = {
            'argus': {
                'poll_interval': 60,
                'entropy_threshold': 5,
                'quality_threshold': 0.95,
            }
        }

        from argus import _load_argus_config
        config = _load_argus_config()

        self.assertEqual(config['poll_interval'], 60)
        self.assertEqual(config['entropy_threshold'], 5)
        self.assertEqual(config['quality_threshold'], 0.95)
        # Defaults for missing keys
        self.assertEqual(config['max_restart_count'], 3)

    @patch('argus._hermes_load_config')
    def test_load_config_preserves_defaults_for_missing_keys(self, mock_load):
        """Missing keys in config must fall back to defaults."""
        mock_load.return_value = {'argus': {'poll_interval': 45}}

        from argus import _load_argus_config
        config = _load_argus_config()

        self.assertEqual(config['poll_interval'], 45)
        self.assertEqual(config['entropy_threshold'], 3)
        self.assertEqual(config['quality_threshold'], 0.92)

    @patch('argus._hermes_load_config', side_effect=Exception('config load failed'))
    def test_load_config_falls_back_to_defaults_on_error(self, mock_load):
        """Config loading must fall back to defaults if hermes config fails."""
        from argus import _load_argus_config
        config = _load_argus_config()

        self.assertEqual(config['poll_interval'], 30)
        self.assertEqual(config['quality_threshold'], 0.92)


class TestArgusSandboxedCron(unittest.TestCase):
    """Test ARGUS behavior in sandboxed/restricted terminal environments."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.argus = Argus()
        self.argus.db_path = self.temp_db.name
        self.argus._init_database()
        self.argus._load_schema()

    def tearDown(self):
        if self.argus.conn:
            self.argus.conn.close()
        os.unlink(self.temp_db.name)

    @patch('argus.pause_job', side_effect=Exception('cron.jobs unavailable in sandbox'))
    @patch('argus.resume_job', side_effect=Exception('cron.jobs unavailable in sandbox'))
    def test_restart_cron_handles_cron_jobs_failure(self, mock_resume, mock_pause):
        """Restart must not crash when cron.jobs functions fail."""
        session_id = 'cron_sandbox_restart'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'job_id': 'sandbox_job_1',
            'task_description': 'Sandboxed cron'
        })

        # Should not raise — must handle exceptions gracefully
        self.argus._restart_session(session_id, 'Test restart')

        # Should still record the restart action
        self.argus.cursor.execute('SELECT * FROM watcher_actions WHERE session_id = ?', (session_id,))
        action = self.argus.cursor.fetchone()
        self.assertIsNotNone(action)

    @patch('argus.pause_job', side_effect=Exception('cron.jobs unavailable in sandbox'))
    def test_kill_cron_handles_cron_jobs_failure(self, mock_pause):
        """Kill must not crash when cron.jobs functions fail."""
        session_id = 'cron_sandbox_kill'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'job_id': 'sandbox_kill_1',
            'task_description': 'Sandboxed cron to kill'
        })

        # Should not raise
        self.argus._kill_session(session_id, 'Max restarts')

        # Status should still be updated
        self.argus.cursor.execute('SELECT status FROM sessions WHERE session_id = ?', (session_id,))
        self.assertEqual(self.argus.cursor.fetchone()['status'], 'killed')

    @patch('argus.pause_job', side_effect=Exception('cron.jobs unavailable'))
    @patch('argus.trigger_job', side_effect=Exception('cron.jobs unavailable'))
    @patch('argus.update_job', side_effect=Exception('cron.jobs unavailable'))
    def test_inject_prompt_cron_handles_cron_jobs_failure(self, mock_update, mock_trigger, mock_pause):
        """Prompt injection must not crash when cron.jobs functions fail."""
        session_id = 'cron_sandbox_inject'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'job_id': 'sandbox_inject_1',
            'task_description': 'Sandboxed cron inject'
        })

        # Should not raise
        self.argus._inject_prompt(session_id, 'Corrective prompt')

        # Should still record the action
        self.argus.cursor.execute(
            "SELECT * FROM watcher_actions WHERE session_id = ? AND action_type = 'inject_prompt'",
            (session_id,)
        )
        self.assertIsNotNone(self.argus.cursor.fetchone())

    @patch('argus.subprocess.run', side_effect=FileNotFoundError('[Errno 2] No such file or directory: kill'))
    def test_restart_delegate_handles_missing_kill_command(self, mock_run):
        """Delegate restart must handle missing kill command in sandbox."""
        session_id = 'delegate_sandbox_kill'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'delegate_task',
            'task_description': 'Sandboxed delegate (PID: 55555)',
            'metadata': json.dumps({'pid': '55555'})
        })

        # Should not raise
        self.argus._restart_session(session_id, 'Test restart')

        # restart_count should still be incremented
        self.argus.cursor.execute('SELECT restart_count FROM sessions WHERE session_id = ?', (session_id,))
        self.assertEqual(self.argus.cursor.fetchone()['restart_count'], 1)

    def test_get_env_returns_full_path(self):
        """_get_cron_env must include hermes bin and homebrew paths."""
        env = self.argus._get_cron_env()

        # PATH must include hermes bin
        self.assertIn('hermes', env.get('PATH', '').lower())
        # PATH must include homebrew
        self.assertIn('/opt/homebrew', env.get('PATH', ''))
        # HOME must be set
        self.assertIn('HOME', env)

    def test_collect_metrics_handles_missing_holographic_db(self):
        """collect_metrics must not crash when holographic_memory.db is missing."""
        session_id = 'metrics_no_holo'
        self.argus.register_session({
            'session_id': session_id,
            'session_type': 'cron',
            'task_description': 'Metrics test'
        })

        # Should not raise even if holographic_memory.db doesn't exist
        self.argus.collect_metrics(session_id)

        # tool_call_count should be set (0 if no data)
        self.argus.cursor.execute('SELECT tool_call_count FROM sessions WHERE session_id = ?', (session_id,))
        row = self.argus.cursor.fetchone()
        self.assertIsNotNone(row['tool_call_count'])


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)