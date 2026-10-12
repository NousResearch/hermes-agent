"""Local control-socket delivery through the profile's live gateway adapter."""
from pathlib import Path
import json


def send_message_verb(runner):
    """Socket permissions provide local-user authorization; homes must be served."""
    def handler(params: dict) -> dict:
        from gateway.config import Platform
        from gateway.run import _profile_runtime_scope
        from tools.send_message_tool import send_message_tool

        target, message, home = params.get("target"), params.get("message"), params.get("home")
        if (not isinstance(target, str) or not target.lower().startswith("matrix:")
                or not isinstance(message, str) or not message.strip() or not isinstance(home, str)):
            return {"error": "send-message requires a Matrix target, message and profile home"}
        requested = Path(home)
        profile = runner._served_profile_for_home(requested)
        if profile is None:
            return {"error": "Profile home is not served by this gateway"}
        with _profile_runtime_scope(requested):
            if runner._authorization_adapter(Platform.MATRIX, profile) is None:
                return {"error": "Gateway Matrix adapter is not connected"}
            args = {"action": "send", "target": target, "message": message}
            if "mentions" in params:
                args["mentions"] = params["mentions"]
            result = send_message_tool(args)
            return json.loads(result) if isinstance(result, str) else result
    return handler
