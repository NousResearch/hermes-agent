"""
UE5 Bridge for External Control.

Communicates with UE5 via Python remote execution plugin or socket.
"""

import logging
import json
import socket
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class UE5Bridge:
    """
    Bridge between Cosmos and Unreal Engine 5.

    Connection modes:
    1. Direct (inside UE5 Editor) - use 'unreal' module
    2. Remote (external) - use Python Remote Execution plugin
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 6789):
        self.host = host
        self.port = port
        self._socket: Optional[socket.socket] = None
        self._connected = False
        self._mode = "remote"

        # Try direct import (only works inside UE5)
        try:
            import unreal
            self._unreal = unreal
            self._mode = "direct"
            logger.info("UE5 Bridge: Direct mode (inside editor)")
        except ImportError:
            self._unreal = None
            logger.info("UE5 Bridge: Remote mode (external)")

    def connect(self) -> bool:
        """Connect to UE5 (for remote mode)."""
        if self._mode == "direct":
            return True

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self.host, self.port))
            self._connected = True
            logger.info(f"Connected to UE5 at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to UE5: {e}")
            return False

    def disconnect(self):
        """Disconnect from UE5."""
        if self._socket:
            self._socket.close()
            self._socket = None
            self._connected = False

    def execute_python(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code in UE5.

        Args:
            code: Python code to execute

        Returns:
            Result dict with 'success' and 'result' or 'error'
        """
        if self._mode == "direct" and self._unreal:
            try:
                # Execute with restricted builtins (AGI v1.9.1)
                from Cosmos.core.safe_eval import SAFE_BUILTINS
                local_vars = {"unreal": self._unreal}
                exec(code, {"__builtins__": SAFE_BUILTINS, "unreal": self._unreal}, local_vars)
                return {"success": True, "result": local_vars}
            except Exception as e:
                return {"success": False, "error": str(e)}

        elif self._connected and self._socket:
            try:
                # Send code to remote executor
                message = json.dumps({"command": "execute", "code": code})
                self._socket.send(message.encode())

                response = self._socket.recv(4096).decode()
                return json.loads(response)
            except Exception as e:
                return {"success": False, "error": str(e)}

        return {"success": False, "error": "Not connected to UE5"}

    # High-level operations

    def spawn_actor(
        self,
        actor_class: str,
        location: tuple,
        rotation: tuple = (0, 0, 0)
    ) -> Dict[str, Any]:
        """Spawn an actor in the current level."""
        code = f"""
actor_class = unreal.load_class(None, "{actor_class}")
location = unreal.Vector({location[0]}, {location[1]}, {location[2]})
rotation = unreal.Rotator({rotation[0]}, {rotation[1]}, {rotation[2]})
actor = unreal.EditorLevelLibrary.spawn_actor_from_class(actor_class, location, rotation)
result = str(actor.get_name()) if actor else None
"""
        return self.execute_python(code)

    def create_material(self, name: str, base_color: tuple) -> Dict[str, Any]:
        """Create a new material."""
        code = f"""
asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
mat_factory = unreal.MaterialFactoryNew()
mat = asset_tools.create_asset("{name}", "/Game/Materials", unreal.Material, mat_factory)
# Note: Setting properties requires material expressions
result = str(mat.get_name()) if mat else None
"""
        return self.execute_python(code)

    def import_assets(self, paths: List[str], destination: str) -> Dict[str, Any]:
        """Import assets from disk."""
        paths_str = str(paths)
        code = f"""
import_data = unreal.AutomatedAssetImportData()
import_data.set_editor_property('filenames', {paths_str})
import_data.set_editor_property('destination_path', "{destination}")
asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
imported = asset_tools.import_assets_automated(import_data)
result = len(imported)
"""
        return self.execute_python(code)

    def take_screenshot(self, output_path: str) -> Dict[str, Any]:
        """Take a viewport screenshot."""
        code = f"""
unreal.AutomationLibrary.take_high_res_screenshot(1920, 1080, "{output_path}")
result = True
"""
        return self.execute_python(code)

    def build_lighting(self) -> Dict[str, Any]:
        """Build lighting for the current level."""
        code = """
unreal.EditorLevelLibrary.build_lighting()
result = True
"""
        return self.execute_python(code)

    def get_selected_actors(self) -> Dict[str, Any]:
        """Get currently selected actors."""
        code = """
selected = unreal.EditorLevelLibrary.get_selected_level_actors()
result = [a.get_name() for a in selected]
"""
        return self.execute_python(code)
