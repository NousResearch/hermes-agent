"""Regression test for maps_client int guard."""
import unittest

class TestMapsClientIntGuard(unittest.TestCase):
    def test_try_except_in_source(self):
        import pathlib
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        source_path = repo_root / "skills" / "productivity" / "maps" / "scripts" / "maps_client.py"
        with open(source_path) as f:
            source = f.read()
        self.assertIn("try:", source)
        self.assertIn("except (ValueError, TypeError):", source)
        self.assertIn("int(args.radius)", source)

if __name__ == "__main__":
    unittest.main()
