import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("check_backend", Path(__file__).parents[1] / "check_backend.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ContractTests(unittest.TestCase):
    def test_decorators_are_collected_without_importing_backend(self):
        source = '''raise RuntimeError("never import the backend")
@method("session.list")
def list_session(rid, params): pass
@router.post("/tasks")
def create_task(payload): pass
@router.put("/jobs/{job_id}")
def update_job(payload): pass
@app.websocket("/api/ws")
async def gateway_ws(socket): pass
'''
        self.assertEqual(module.surface(source), {("rpc", "session.list"), ("post", "/tasks"), ("put", "/jobs/{job_id}"), ("websocket", "/api/ws")})

    def test_removed_route_is_reported_as_incompatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "routes.py").write_text('@router.get("/board")\ndef board(): pass\n')
            missing = module.check_requirements(root, {"routes.py": [["get", "/board"], ["post", "/tasks"]]})
            self.assertEqual(missing, ["routes.py: missing post /tasks"])

    def test_missing_source_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(module.check_requirements(Path(tmp), {"absent.py": [["rpc", "ping"]]}))

if __name__ == "__main__":
    unittest.main()
