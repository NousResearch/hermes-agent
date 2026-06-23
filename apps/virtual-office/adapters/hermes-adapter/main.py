import json
import os
import sys


def hermes_root() -> str:
    return os.path.expanduser("~/.hermes")


def sessions_index_path() -> str:
    return os.path.join(hermes_root(), "sessions", "sessions.json")


def session_file_path(session_id: str) -> str:
    return os.path.join(hermes_root(), "sessions", f"session_{session_id}.json")


def load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def list_sessions() -> list[dict]:
    try:
        sessions = load_json_file(sessions_index_path())
    except FileNotFoundError:
        return []

    if not isinstance(sessions, dict):
        return []

    summaries: list[dict] = []
    for session_id, session_data in sessions.items():
        session_data = session_data if isinstance(session_data, dict) else {}
        summaries.append(
            {
                "id": session_id,
                "display_name": session_data.get("display_name"),
                "platform": session_data.get("platform"),
                "chat_type": session_data.get("chat_type"),
                "created_at": session_data.get("created_at"),
                "updated_at": session_data.get("updated_at"),
                "total_tokens": session_data.get("total_tokens"),
                "estimated_cost_usd": session_data.get("estimated_cost_usd"),
            }
        )

    return summaries


def resume_session(session_id: str) -> dict:
    return load_json_file(session_file_path(session_id))


def handle_request(payload: dict) -> dict:
    request_id = payload.get("id", 1)
    method = payload.get("method")
    params = payload.get("params") or {}

    if method == "ping":
        return {"jsonrpc": "2.0", "result": "pong", "id": request_id}

    if method == "session.list":
        return {"jsonrpc": "2.0", "result": list_sessions(), "id": request_id}

    if method == "session.resume":
        session_id = params.get("session_id")
        if not session_id:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32602, "message": "Missing session_id"},
                "id": request_id,
            }

        try:
            session = resume_session(str(session_id))
        except FileNotFoundError:
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32004, "message": "Session not found"},
                "id": request_id,
            }

        return {"jsonrpc": "2.0", "result": session, "id": request_id}

    return {
        "jsonrpc": "2.0",
        "error": {"code": -32601, "message": "Method not found"},
        "id": request_id,
    }


def main() -> None:
    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
            response = handle_request(payload)
        except json.JSONDecodeError:
            response = {
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": "Parse error"},
                "id": None,
            }
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
