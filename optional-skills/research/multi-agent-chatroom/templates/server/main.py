# server/main.py
"""FastAPI WebSocket chatroom server."""

import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from server.channel import ChannelManager
from server.models import Message
from server.config import load_config, get_channels

app = FastAPI(title="Multi-Agent Chatroom — AI2050")
channel_manager = ChannelManager()


@app.on_event("startup")
async def startup():
    config = load_config()
    for ch_name in get_channels(config).values():
        await channel_manager.create_channel(ch_name)


@app.websocket("/ws/{client_name}")
async def websocket_endpoint(websocket: WebSocket, client_name: str):
    await websocket.accept()
    subscriptions = {}

    async def forward_to_client(message: dict):
        try:
            await websocket.send_text(json.dumps(message, ensure_ascii=False))
        except Exception:
            pass

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            action = data.get("action")

            if action == "subscribe":
                channel = data["channel"]
                # Unsubscribe old handler if re-subscribing
                if channel in subscriptions:
                    await channel_manager.unsubscribe(channel, subscriptions[channel])
                subscriptions[channel] = forward_to_client
                await channel_manager.subscribe(channel, forward_to_client)

                # Send channel history
                history = await channel_manager.get_history(channel)
                for msg in history:
                    await websocket.send_text(json.dumps(msg, ensure_ascii=False))

                await websocket.send_text(json.dumps({
                    "type": "system", "content": f"Subscribed to {channel}"
                }, ensure_ascii=False))

            elif action == "publish":
                channel = data["channel"]
                msg = Message(
                    channel=channel,
                    sender=client_name,
                    content=data["content"],
                    msg_type=data.get("msg_type", "message"),
                    metadata=data.get("metadata", {})
                )
                await channel_manager.publish(channel, msg.to_json())

                # Echo confirmation
                await websocket.send_text(json.dumps({
                    "type": "system", "content": f"Published to {channel}"
                }, ensure_ascii=False))

            elif action == "history":
                channel = data["channel"]
                limit = data.get("limit", 50)
                history = await channel_manager.get_history(channel, limit)
                await websocket.send_text(json.dumps({
                    "type": "history", "channel": channel, "messages": history
                }, ensure_ascii=False))

    except WebSocketDisconnect:
        for channel, handler in subscriptions.items():
            await channel_manager.unsubscribe(channel, handler)


@app.get("/health")
async def health():
    return {"status": "ok", "channels": list(channel_manager._subscribers.keys())}


def run_server():
    import uvicorn
    config = get_server_config()
    uvicorn.run(app, host=config["host"], port=config["port"])


if __name__ == "__main__":
    run_server()
