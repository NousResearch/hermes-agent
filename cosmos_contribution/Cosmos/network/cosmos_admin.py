"""
COSMOS Admin Routes — Web dashboard for managing the COSMOS mesh.

Endpoints:
  GET  /api/cosmos/status         — Mesh status (nodes, bots, mesh state)
  GET  /api/cosmos/bots           — All available bots (local + remote)
  GET  /api/cosmos/pending        — Pending PRO join requests
  POST /api/cosmos/approve/<id>   — Approve a PRO join request
  POST /api/cosmos/reject/<id>    — Reject a PRO join request
  POST /api/cosmos/query          — Query a bot via COSMOS (for testing)
"""
import json
from typing import Optional
from loguru import logger


def register_cosmos_routes(app):
    """
    Register COSMOS admin API routes with the Flask/Quart app.

    Call this during web server setup.
    """
    from .cosmos_node import get_cosmos_node
    from .cosmos_client import get_cosmos_client

    @app.route("/api/cosmos/status", methods=["GET"])
    async def cosmos_status():
        node = get_cosmos_node()
        if not node:
            return json.dumps({"error": "COSMOS node not running"}), 503
        return json.dumps(node.get_status(), indent=2)

    @app.route("/api/cosmos/bots", methods=["GET"])
    async def cosmos_bots():
        node = get_cosmos_node()
        if not node:
            return json.dumps({"error": "COSMOS node not running"}), 503
        bots = node.get_all_bots()
        return json.dumps({
            "bots": [{"name": name, "node": loc} for name, loc in bots.items()],
            "total": len(bots),
            "local": node.get_local_bots(),
        }, indent=2)

    @app.route("/api/cosmos/pending", methods=["GET"])
    async def cosmos_pending():
        node = get_cosmos_node()
        if not node:
            return json.dumps({"error": "COSMOS node not running"}), 503
        pending = [a for a in node._pending_approvals if a["status"] == "pending"]
        return json.dumps({"pending": pending, "count": len(pending)}, indent=2)

    @app.route("/api/cosmos/approve/<request_id>", methods=["POST"])
    async def cosmos_approve(request_id: str):
        node = get_cosmos_node()
        if not node:
            return json.dumps({"error": "COSMOS node not running"}), 503
        if node.approve_join(request_id):
            return json.dumps({"status": "approved", "id": request_id})
        return json.dumps({"error": "Request not found or already processed"}), 404

    @app.route("/api/cosmos/reject/<request_id>", methods=["POST"])
    async def cosmos_reject(request_id: str):
        node = get_cosmos_node()
        if not node:
            return json.dumps({"error": "COSMOS node not running"}), 503
        # Get reason from request body
        try:
            from quart import request
            data = await request.get_json()
            reason = data.get("reason", "Rejected by admin") if data else "Rejected by admin"
        except Exception:
            reason = "Rejected by admin"
        if node.reject_join(request_id, reason):
            return json.dumps({"status": "rejected", "id": request_id})
        return json.dumps({"error": "Request not found or already processed"}), 404

    @app.route("/api/cosmos/query", methods=["POST"])
    async def cosmos_query():
        """Test endpoint — query a bot through COSMOS."""
        try:
            from quart import request
            data = await request.get_json()
        except Exception:
            return json.dumps({"error": "Invalid JSON body"}), 400

        bot_name = data.get("bot")
        prompt = data.get("prompt")
        if not bot_name or not prompt:
            return json.dumps({"error": "Missing 'bot' or 'prompt'"}), 400

        client = get_cosmos_client()
        response = await client.query(bot_name, prompt, max_tokens=data.get("max_tokens", 4000))

        if response:
            return json.dumps({"bot": bot_name, "response": response})
        return json.dumps({"error": f"Bot '{bot_name}' unavailable or no response"}), 503

    logger.info("COSMOS admin routes registered")
