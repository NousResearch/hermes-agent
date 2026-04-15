import json
import subprocess
import sys
import textwrap
from pathlib import Path


def test_image_gen_is_not_registered_when_fal_client_import_fails():
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import importlib.abc
        import json
        import sys

        sys.path.insert(0, ".")

        class BlockFal(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "fal_client" or fullname.startswith("fal_client."):
                    raise ImportError("blocked fal_client for test")
                return None

        sys.meta_path.insert(0, BlockFal())

        import model_tools

        print(json.dumps({
            "toolset_registered": "image_gen" in model_tools.TOOLSET_REQUIREMENTS,
            "tool_registered": "image_generate" in model_tools.TOOL_TO_TOOLSET_MAP,
            "unavailable": [item["name"] for item in model_tools.check_tool_availability()[1]],
        }))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["toolset_registered"] is False
    assert payload["tool_registered"] is False
    assert "image_gen" not in payload["unavailable"]
