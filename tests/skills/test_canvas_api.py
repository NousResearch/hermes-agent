"""Tests for the Canvas LMS API CLI (canvas_api.py)."""

import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add the canvas scripts directory to the path so we can import canvas_api
CANVAS_SCRIPT_DIR = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    "skills",
    "productivity",
    "canvas",
    "scripts",
)
sys.path.insert(0, os.path.abspath(CANVAS_SCRIPT_DIR))

import canvas_api


def _make_args(**kwargs):
    """Build an argparse-like Namespace for testing."""
    ns = MagicMock()
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


def _mock_response(json_data, status_code=200, headers=None):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = json.dumps(json_data)
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# list_courses
# ---------------------------------------------------------------------------
class TestListCourses(unittest.TestCase):
    @patch.object(canvas_api, "CANVAS_API_TOKEN", "tok")
    @patch.object(canvas_api, "CANVAS_BASE_URL", "https://canvas.test")
    @patch("canvas_api.requests")
    def test_list_courses_success(self, mock_requests):
        courses = [
            {
                "id": 1,
                "name": "Math 101",
                "course_code": "MATH101",
                "enrollment_term_id": 10,
                "start_at": "2025-01-01",
                "end_at": "2025-06-01",
                "workflow_state": "available",
            }
        ]
        mock_requests.get.return_value = _mock_response(courses)
        mock_requests.HTTPError = Exception

        args = _make_args(per_page=50, enrollment_state="active")
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            canvas_api.list_courses(args)
        output = json.loads(buf.getvalue())
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["id"], 1)
        self.assertEqual(output[0]["name"], "Math 101")


# ---------------------------------------------------------------------------
# list_assignments
# ---------------------------------------------------------------------------
class TestListAssignments(unittest.TestCase):
    @patch.object(canvas_api, "CANVAS_API_TOKEN", "tok")
    @patch.object(canvas_api, "CANVAS_BASE_URL", "https://canvas.test")
    @patch("canvas_api.requests")
    def test_list_assignments_success(self, mock_requests):
        assignments = [
            {
                "id": 10,
                "name": "HW 1",
                "description": "Do things",
                "due_at": "2025-02-01",
                "points_possible": 50,
                "submission_types": ["online_text_entry"],
                "html_url": "https://canvas.test/a/10",
                "course_id": 1,
            }
        ]
        mock_requests.get.return_value = _mock_response(assignments)
        mock_requests.HTTPError = Exception

        args = _make_args(course_id="1", per_page=50, order_by="due_at")
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            canvas_api.list_assignments(args)
        output = json.loads(buf.getvalue())
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["name"], "HW 1")


# ---------------------------------------------------------------------------
# get_assignment
# ---------------------------------------------------------------------------
class TestGetAssignment(unittest.TestCase):
    @patch.object(canvas_api, "CANVAS_API_TOKEN", "tok")
    @patch.object(canvas_api, "CANVAS_BASE_URL", "https://canvas.test")
    @patch("canvas_api.requests")
    def test_get_assignment_success(self, mock_requests):
        assignment = {
            "id": 10,
            "name": "Final Project",
            "course_id": 1,
            "description": "<p>Full description here</p>",
            "due_at": "2025-05-01",
            "points_possible": 100,
            "submission_types": ["online_upload"],
            "html_url": "https://canvas.test/a/10",
            "attachments": [
                {
                    "display_name": "rubric.pdf",
                    "url": "https://canvas.test/files/1/download",
                    "content-type": "application/pdf",
                    "size": 204800,
                }
            ],
            "external_tool_tag_attributes": None,
            "locked_for_user": False,
            "lock_explanation": "",
        }
        mock_requests.get.return_value = _mock_response(assignment)
        mock_requests.HTTPError = Exception

        args = _make_args(course_id="1", assignment_id="10")
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            canvas_api.get_assignment(args)
        output = json.loads(buf.getvalue())
        self.assertEqual(output["id"], 10)
        self.assertEqual(output["description"], "<p>Full description here</p>")
        self.assertEqual(len(output["attachments"]), 1)
        self.assertEqual(output["attachments"][0]["display_name"], "rubric.pdf")
        self.assertFalse(output["google_assignments"])
        self.assertEqual(output["google_assignments_url"], "")

    @patch.object(canvas_api, "CANVAS_API_TOKEN", "tok")
    @patch.object(canvas_api, "CANVAS_BASE_URL", "https://canvas.test")
    @patch("canvas_api.requests")
    def test_get_assignment_google_detected(self, mock_requests):
        assignment = {
            "id": 20,
            "name": "Google Essay",
            "course_id": 1,
            "description": "",
            "due_at": None,
            "points_possible": 50,
            "submission_types": ["external_tool"],
            "html_url": "https://canvas.test/a/20",
            "attachments": None,
            "external_tool_tag_attributes": {
                "url": "https://assignments.google.com/v1/courses/abc123"
            },
            "locked_for_user": False,
            "lock_explanation": "",
        }
        mock_requests.get.return_value = _mock_response(assignment)
        mock_requests.HTTPError = Exception

        args = _make_args(course_id="1", assignment_id="20")
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            canvas_api.get_assignment(args)
        output = json.loads(buf.getvalue())
        self.assertTrue(output["google_assignments"])
        self.assertIn("google", output["google_assignments_url"])


# ---------------------------------------------------------------------------
# submit_assignment
# ---------------------------------------------------------------------------
class TestSubmitAssignment(unittest.TestCase):
    def _make_assignment_response(self, submission_types):
        return {
            "id": 10,
            "name": "HW 1",
            "submission_types": submission_types,
            "html_url": "https://canvas.test/a/10",
            "external_tool_tag_attributes": None,
        }

    @patch.object(canvas_api, "CANVAS_API_TOKEN", "tok")
    @patch.object(canvas_api, "CANVAS_BASE_URL", "https://canvas.test")
    @patch("canvas_api.requests")
    def test_submit_text_entry_dry_run(self, mock_requests):
        mock_requests.get.return_value = _mock_response(
            self._make_assignment_response(["online_text_entry"])
        )
        mock_requests.HTTPError = Exception

        args = _make_args(
            course_id="1",
            assignment_id="10",
            type="online_text_entry",
            body="My answer",
            url=None,
            file=None,
            dry_run=True,
        )
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            canvas_api.submit_assignment(args)
        output = json.loads(buf.getvalue())
        self.assertTrue(output["dry_run"])
        self.assertEqual(output["submission_type"], "online_text_entry")
        self.assertEqual(output["body"], "My answer")
        # Verify no POST was made (dry run)
        mock_requests.post.assert_not_called()

    @patch.object(canvas_api, "CANVAS_API_TOKEN", "tok")
    @patch.object(canvas_api, "CANVAS_BASE_URL", "https://canvas.test")
    @patch("canvas_api.requests")
    def test_submit_text_entry_success(self, mock_requests):
        mock_requests.get.return_value = _mock_response(
            self._make_assignment_response(["online_text_entry"])
        )
        mock_requests.post.return_value = _mock_response(
            {
                "id": 999,
                "submitted_at": "2025-04-30T18:00:00Z",
                "workflow_state": "submitted",
                "submission_type": "online_text_entry",
            }
        )
        mock_requests.HTTPError = Exception

        args = _make_args(
            course_id="1",
            assignment_id="10",
            type="online_text_entry",
            body="Done",
            url=None,
            file=None,
            dry_run=False,
        )
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            canvas_api.submit_assignment(args)
        output = json.loads(buf.getvalue())
        self.assertTrue(output["success"])
        self.assertEqual(output["submission_id"], 999)

    @patch.object(canvas_api, "CANVAS_API_TOKEN", "tok")
    @patch.object(canvas_api, "CANVAS_BASE_URL", "https://canvas.test")
    def test_submit_url_missing(self):
        args = _make_args(
            course_id="1",
            assignment_id="10",
            type="online_url",
            body="",
            url=None,
            file=None,
            dry_run=False,
        )
        with self.assertRaises(SystemExit):
            canvas_api.submit_assignment(args)

    @patch.object(canvas_api, "CANVAS_API_TOKEN", "tok")
    @patch.object(canvas_api, "CANVAS_BASE_URL", "https://canvas.test")
    def test_submit_file_missing(self):
        args = _make_args(
            course_id="1",
            assignment_id="10",
            type="online_upload",
            body="",
            url=None,
            file=None,
            dry_run=False,
        )
        with self.assertRaises(SystemExit):
            canvas_api.submit_assignment(args)

    @patch.object(canvas_api, "CANVAS_API_TOKEN", "tok")
    @patch.object(canvas_api, "CANVAS_BASE_URL", "https://canvas.test")
    def test_submit_file_not_found(self):
        args = _make_args(
            course_id="1",
            assignment_id="10",
            type="online_upload",
            body="",
            url=None,
            file="/nonexistent/path/hw.pdf",
            dry_run=False,
        )
        with self.assertRaises(SystemExit):
            canvas_api.submit_assignment(args)

    @patch.object(canvas_api, "CANVAS_API_TOKEN", "tok")
    @patch.object(canvas_api, "CANVAS_BASE_URL", "https://canvas.test")
    @patch("canvas_api.requests")
    def test_submit_type_not_allowed(self, mock_requests):
        mock_requests.get.return_value = _mock_response(
            self._make_assignment_response(["online_upload"])
        )
        mock_requests.HTTPError = Exception

        args = _make_args(
            course_id="1",
            assignment_id="10",
            type="online_text_entry",
            body="text",
            url=None,
            file=None,
            dry_run=False,
        )
        buf = io.StringIO()
        with self.assertRaises(SystemExit):
            with patch("sys.stdout", buf):
                canvas_api.submit_assignment(args)
        output = json.loads(buf.getvalue())
        self.assertEqual(output["error"], "submission_type_not_allowed")

    @patch.object(canvas_api, "CANVAS_API_TOKEN", "tok")
    @patch.object(canvas_api, "CANVAS_BASE_URL", "https://canvas.test")
    @patch("canvas_api.requests")
    def test_submit_google_assignments_error(self, mock_requests):
        assignment = {
            "id": 10,
            "name": "Google Essay",
            "submission_types": ["external_tool"],
            "html_url": "https://canvas.test/a/10",
            "external_tool_tag_attributes": {
                "url": "https://assignments.google.com/v1/abc"
            },
        }
        mock_requests.get.return_value = _mock_response(assignment)
        mock_requests.HTTPError = Exception

        args = _make_args(
            course_id="1",
            assignment_id="10",
            type="online_text_entry",
            body="text",
            url=None,
            file=None,
            dry_run=False,
        )
        buf = io.StringIO()
        with self.assertRaises(SystemExit):
            with patch("sys.stdout", buf):
                canvas_api.submit_assignment(args)
        output = json.loads(buf.getvalue())
        self.assertEqual(output["error"], "google_assignments")
        self.assertIn("google", output["google_assignments_url"])

    @patch.object(canvas_api, "CANVAS_API_TOKEN", "tok")
    @patch.object(canvas_api, "CANVAS_BASE_URL", "https://canvas.test")
    @patch("canvas_api.requests")
    def test_submit_no_submission_type(self, mock_requests):
        mock_requests.get.return_value = _mock_response(
            self._make_assignment_response(["no_submission"])
        )
        mock_requests.HTTPError = Exception

        args = _make_args(
            course_id="1",
            assignment_id="10",
            type="online_text_entry",
            body="text",
            url=None,
            file=None,
            dry_run=False,
        )
        buf = io.StringIO()
        with self.assertRaises(SystemExit):
            with patch("sys.stdout", buf):
                canvas_api.submit_assignment(args)
        output = json.loads(buf.getvalue())
        self.assertEqual(output["error"], "submission_type_not_allowed")

    @patch.object(canvas_api, "CANVAS_API_TOKEN", "tok")
    @patch.object(canvas_api, "CANVAS_BASE_URL", "https://canvas.test")
    @patch("canvas_api.requests")
    def test_submit_file_upload_flow(self, mock_requests):
        # Create a temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"fake pdf content")
            tmp_path = f.name

        try:
            # GET assignment
            mock_requests.get.return_value = _mock_response(
                self._make_assignment_response(["online_upload"])
            )
            # POST calls: upload slot, file upload, final submission
            mock_requests.post.side_effect = [
                _mock_response(
                    {
                        "upload_url": "https://canvas.test/upload/slot",
                        "upload_params": {"param1": "val1"},
                    }
                ),
                _mock_response({"id": 555}),
                _mock_response(
                    {
                        "id": 999,
                        "submitted_at": "2025-04-30T18:00:00Z",
                        "workflow_state": "submitted",
                        "submission_type": "online_upload",
                    }
                ),
            ]
            mock_requests.HTTPError = Exception

            args = _make_args(
                course_id="1",
                assignment_id="10",
                type="online_upload",
                body="",
                url=None,
                file=tmp_path,
                dry_run=False,
            )
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                canvas_api.submit_assignment(args)
            output = json.loads(buf.getvalue())
            self.assertTrue(output["success"])
            self.assertEqual(output["submission_id"], 999)
            self.assertEqual(mock_requests.post.call_count, 3)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# sync_assignments
# ---------------------------------------------------------------------------
class TestSyncAssignments(unittest.TestCase):
    @patch.object(canvas_api, "CANVAS_API_TOKEN", "tok")
    @patch.object(canvas_api, "CANVAS_BASE_URL", "https://canvas.test")
    @patch("canvas_api.requests")
    def test_sync_assignments(self, mock_requests):
        assignments = [
            {
                "id": 10,
                "course_id": 1,
                "name": "HW 1",
                "due_at": "2025-02-01",
                "points_possible": 50,
                "submission_types": ["online_text_entry"],
                "html_url": "https://canvas.test/a/10",
            },
            {
                "id": 20,
                "course_id": 1,
                "name": "HW 2",
                "due_at": "2025-03-01",
                "points_possible": 100,
                "submission_types": ["online_upload"],
                "html_url": "https://canvas.test/a/20",
            },
        ]
        mock_requests.get.return_value = _mock_response(assignments)
        mock_requests.HTTPError = Exception

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(canvas_api, "_db_path", return_value=os.path.join(tmpdir, "test.db")):
                args = _make_args(course_id="1", per_page=50)
                buf = io.StringIO()
                with patch("sys.stdout", buf):
                    canvas_api.sync_assignments(args)
                output = json.loads(buf.getvalue())
                self.assertEqual(output["synced"], 2)
                self.assertEqual(output["course_id"], 1)

                # Verify DB contents
                conn = sqlite3.connect(os.path.join(tmpdir, "test.db"))
                conn.row_factory = sqlite3.Row
                rows = conn.execute("SELECT * FROM assignments ORDER BY id").fetchall()
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["name"], "HW 1")
                self.assertEqual(rows[1]["name"], "HW 2")
                # Verify local_done is 0 by default
                self.assertEqual(rows[0]["local_done"], 0)
                conn.close()


# ---------------------------------------------------------------------------
# mark_done
# ---------------------------------------------------------------------------
class TestMarkDone(unittest.TestCase):
    def test_mark_done_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with patch.object(canvas_api, "_db_path", return_value=db_path):
                # Seed a row
                conn = canvas_api._get_db()
                conn.execute(
                    "INSERT INTO assignments (id, course_id, name, last_synced) VALUES (?, ?, ?, ?)",
                    (10, 1, "HW 1", "2025-01-01"),
                )
                conn.commit()
                conn.close()

                args = _make_args(course_id="1", assignment_id="10", notes="All done")
                buf = io.StringIO()
                with patch("sys.stdout", buf):
                    canvas_api.mark_done(args)
                output = json.loads(buf.getvalue())
                self.assertTrue(output["success"])
                self.assertEqual(output["assignment_id"], 10)

                # Verify DB
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM assignments WHERE id=10").fetchone()
                self.assertEqual(row["local_done"], 1)
                self.assertEqual(row["done_notes"], "All done")
                conn.close()

    def test_mark_done_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with patch.object(canvas_api, "_db_path", return_value=db_path):
                # Initialize the DB but don't seed any rows
                conn = canvas_api._get_db()
                conn.close()

                args = _make_args(course_id="1", assignment_id="999", notes="")
                with self.assertRaises(SystemExit):
                    canvas_api.mark_done(args)


# ---------------------------------------------------------------------------
# list_pending / list_done
# ---------------------------------------------------------------------------
class TestListPendingAndDone(unittest.TestCase):
    def _seed_db(self, db_path):
        with patch.object(canvas_api, "_db_path", return_value=db_path):
            conn = canvas_api._get_db()
            conn.execute(
                "INSERT INTO assignments (id, course_id, name, due_at, submission_types, last_synced, local_done) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (10, 1, "HW 1", "2025-02-01", '["online_text_entry"]', "2025-01-01", 0),
            )
            conn.execute(
                "INSERT INTO assignments (id, course_id, name, due_at, submission_types, last_synced, local_done, done_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (20, 1, "HW 2", "2025-03-01", '["online_upload"]', "2025-01-01", 1, "2025-02-28"),
            )
            conn.execute(
                "INSERT INTO assignments (id, course_id, name, due_at, submission_types, last_synced, local_done) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (30, 2, "Quiz 1", "2025-04-01", '["no_submission"]', "2025-01-01", 0),
            )
            conn.commit()
            conn.close()

    def test_list_pending_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            self._seed_db(db_path)
            with patch.object(canvas_api, "_db_path", return_value=db_path):
                args = _make_args(course_id=None)
                buf = io.StringIO()
                with patch("sys.stdout", buf):
                    canvas_api.list_pending(args)
                output = json.loads(buf.getvalue())
                self.assertEqual(len(output), 2)  # HW 1 and Quiz 1
                self.assertEqual(output[0]["name"], "HW 1")
                self.assertEqual(output[1]["name"], "Quiz 1")

    def test_list_pending_by_course(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            self._seed_db(db_path)
            with patch.object(canvas_api, "_db_path", return_value=db_path):
                args = _make_args(course_id="1")
                buf = io.StringIO()
                with patch("sys.stdout", buf):
                    canvas_api.list_pending(args)
                output = json.loads(buf.getvalue())
                self.assertEqual(len(output), 1)
                self.assertEqual(output[0]["name"], "HW 1")

    def test_list_done_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            self._seed_db(db_path)
            with patch.object(canvas_api, "_db_path", return_value=db_path):
                args = _make_args(course_id=None)
                buf = io.StringIO()
                with patch("sys.stdout", buf):
                    canvas_api.list_done(args)
                output = json.loads(buf.getvalue())
                self.assertEqual(len(output), 1)
                self.assertEqual(output[0]["name"], "HW 2")

    def test_list_done_by_course(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            self._seed_db(db_path)
            with patch.object(canvas_api, "_db_path", return_value=db_path):
                args = _make_args(course_id="2")
                buf = io.StringIO()
                with patch("sys.stdout", buf):
                    canvas_api.list_done(args)
                output = json.loads(buf.getvalue())
                self.assertEqual(len(output), 0)


# ---------------------------------------------------------------------------
# sync preserves local_done
# ---------------------------------------------------------------------------
class TestSyncPreservesLocalDone(unittest.TestCase):
    @patch.object(canvas_api, "CANVAS_API_TOKEN", "tok")
    @patch.object(canvas_api, "CANVAS_BASE_URL", "https://canvas.test")
    @patch("canvas_api.requests")
    def test_sync_does_not_overwrite_done(self, mock_requests):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with patch.object(canvas_api, "_db_path", return_value=db_path):
                # Seed with a done assignment
                conn = canvas_api._get_db()
                conn.execute(
                    "INSERT INTO assignments (id, course_id, name, last_synced, local_done, done_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (10, 1, "HW 1", "2025-01-01", 1, "2025-02-01"),
                )
                conn.commit()
                conn.close()

                # Sync with updated data from Canvas
                assignments = [
                    {
                        "id": 10,
                        "course_id": 1,
                        "name": "HW 1 (updated)",
                        "due_at": "2025-02-15",
                        "points_possible": 75,
                        "submission_types": ["online_text_entry"],
                        "html_url": "https://canvas.test/a/10",
                    }
                ]
                mock_requests.get.return_value = _mock_response(assignments)
                mock_requests.HTTPError = Exception

                args = _make_args(course_id="1", per_page=50)
                buf = io.StringIO()
                with patch("sys.stdout", buf):
                    canvas_api.sync_assignments(args)

                # Verify name was updated but local_done preserved
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                row = conn.execute("SELECT * FROM assignments WHERE id=10").fetchone()
                self.assertEqual(row["name"], "HW 1 (updated)")
                self.assertEqual(row["local_done"], 1)  # preserved!
                self.assertEqual(row["done_at"], "2025-02-01")  # preserved!
                conn.close()


if __name__ == "__main__":
    unittest.main()
