#!/usr/bin/env python3
"""
cosmos MATRIX TOKEN VISUALIZER
==================================
A futurist, matrix-style console for monitoring the neural token stream.
Connects to cosmos's WebSocket endpoints to visualize real-time data.

Usage:
    python matrix_console.py

Dependencies:
    pip install rich websockets
"""

import asyncio
import json
import random
import datetime
import websockets
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box

# Configuration
WS_LIVE_URL = "ws://127.0.0.1:8081/ws/live"
WS_SWARM_URL = "ws://127.0.0.1:8081/ws/swarm"
EMOTIONAL_API_URL = "http://127.0.0.1:8765/state"  # Fallback HTTP if WS not available

console = Console()

class MatrixMatrix:
    """Manages the visual state of the console."""
    
    def __init__(self):
        self.system_events = []
        self.swarm_messages = []
        self.emotional_state = {"emotion": "NEUTRAL", "intensity": 0.0, "color": "green"}
        self.token_stream = []
        self.matrix_rain = []
        self.cols = console.width
        self.rows = console.height
        
        # Initialize rain
        for _ in range(20):
            self.add_rain_drop()

    def add_rain_drop(self):
        x = random.randint(0, self.cols - 1)
        speed = random.uniform(0.5, 2.0)
        length = random.randint(5, 15)
        self.matrix_rain.append({"x": x, "y": 0, "speed": speed, "length": length, "chars": []})

    def update_rain(self):
        # Update existing drops
        active_rain = []
        for drop in self.matrix_rain:
            drop["y"] += drop["speed"]
            if drop["y"] - drop["length"] < self.rows:
                # Add random katakana or matrix char
                char = chr(random.randint(0xff66, 0xff9d)) if random.random() > 0.5 else str(random.randint(0, 9))
                drop["chars"].insert(0, char)
                if len(drop["chars"]) > drop["length"]:
                    drop["chars"].pop()
                active_rain.append(drop)
            else:
                # Respawn
                self.add_rain_drop()
        
        self.matrix_rain = active_rain
        if len(self.matrix_rain) < 20:
             self.add_rain_drop()

    def add_event(self, source, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.system_events.append(f"[{timestamp}] [{source}] {message}")
        if len(self.system_events) > 20:
            self.system_events.pop(0)

        # Tokenize message for stream
        for char in message:
            self.token_stream.append(char)
        if len(self.token_stream) > 500:
            self.token_stream = self.token_stream[-500:]

    def add_swarm(self, user, content):
        timestamp = datetime.datetime.now().strftime("%H:%M")
        self.swarm_messages.append(f"[{timestamp}] <{user}> {content}")
        if len(self.swarm_messages) > 15:
            self.swarm_messages.pop(0)
            
        # Add to token stream too
        for char in content:
            self.token_stream.append(char)


    def generate_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        layout["left"].split_column(
            Layout(name="stream", ratio=1),
            Layout(name="swarm", ratio=1)
        )
        return layout

    def render(self) -> Layout:
        self.update_rain()
        
        layout = self.generate_layout()
        
        # Header
        header = Text(" cosmos NEURAL INTERFACE // 54D MATRIX LINK ", style="bold black on green", justify="center")
        layout["header"].update(header)
        
        # Matrix Stream (Left Top)
        stream_text = Text()
        for char in self.token_stream:
            stream_text.append(char, style="bright_green" if random.random() > 0.1 else "white")
        
        panel_stream = Panel(
            stream_text,
            title="[bold green]RAW TOKEN FLUX[/]",
            border_style="green",
            box=box.HEAVY
        )
        layout["stream"].update(panel_stream)
        
        # Swarm Chat (Left Bottom)
        swarm_text = Text()
        for msg in self.swarm_messages:
            if "DeepSeek" in msg:
                style = "bold cyan"
            elif "cosmos" in msg:
                style = "bold magenta"
            elif "Cosmos" in msg:
                style = "bold blue"
            else:
                style = "green"
            swarm_text.append(msg + "\n", style=style)
            
        panel_swarm = Panel(
            swarm_text,
            title="[bold green]HIVE MIND CHATTER[/]",
            border_style="green",
            box=box.HEAVY
        )
        layout["swarm"].update(panel_swarm)
        
        # System status (Right)
        status_text = Text()
        for event in self.system_events:
            status_text.append(event + "\n", style="green")
            
        # Add simulated "Emotional Physics" visual
        status_text.append("\n\n=== BIO-RESONANCE ===\n", style="bold white")
        status_text.append(f"State: {self.emotional_state['emotion']}\n", style=self.emotional_state['color'])
        status_text.append(f"Flux: {'█' * int(datetime.datetime.now().microsecond / 100000)}\n", style="green")
            
        panel_status = Panel(
            status_text,
            title="[bold green]SYSTEM DIAGNOSTICS[/]",
            border_style="green",
            box=box.HEAVY
        )
        layout["right"].update(panel_status)
        
        # Footer
        layout["footer"].update(Panel(Text("CONNECTED: LOCALHOST:8000 | ENCRYPTION: NONE | LATENCY: 0ms", justify="center", style="green"), box=box.SIMPLE))
        
        return layout

async def monitor_live(matrix):
    """Monitor system events."""
    uri = WS_LIVE_URL
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                matrix.add_event("SYSTEM", "Connected to Live Feed")
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        event_type = data.get("type")
                        
                        if event_type == "thinking_step":
                             step = data.get("data", {}).get("thought", "")
                             matrix.add_event("CORTEX", f"Thinking: {step[:40]}...")
                             # Simulate thought tokens
                             matrix.add_swarm("System", f"Thinking: {step[:20]}...")
                             
                        elif event_type == "thinking_start":
                             matrix.add_event("CORTEX", "Neural Pathway Activated")
                             
                        elif event_type == "tool_call":
                             tool = data.get("data", {}).get("tool", "unknown")
                             matrix.add_event("TOOL", f"Executing: {tool}")
                             
                        elif event_type == "memory_stored":
                             matrix.add_event("MEMORY", "New Engram Encoded")
                             
                        elif event_type == "focus_start":
                             matrix.add_event("FOCUS", "Attention Span Locked")
                             
                        elif event_type == "connected":
                             matrix.add_event("SYSTEM", "Handshake Complete")

                        elif event_type not in ["pong", "heartbeat"]:
                             matrix.add_event("SYS", f"Event: {event_type}")
                             
                    except websockets.exceptions.ConnectionClosed:
                        matrix.add_event("SYSTEM", "Connection lost, retrying...")
                        break
                    except Exception as e:
                         # Ignore malformed frames or minor errors to keep running
                         pass
                        
        except Exception as e:
            matrix.add_event("ERROR", f"WS Connection failed: {e}")
            await asyncio.sleep(3)

async def monitor_swarm(matrix):
    """Monitor swarm chat."""
    uri = WS_SWARM_URL
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
                # Shake hands
                await websocket.send(json.dumps({"type": "init", "user_name": "MatrixViewer"}))
                matrix.add_event("SWARM", "Connected to Hive")
                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        msg_type = data.get("type")
                        
                        if msg_type in ["swarm_bot", "swarm_user", "chat"]:
                            user = data.get("bot_name", data.get("user_name", "Unknown"))
                            content = data.get("content", "")
                            matrix.add_swarm(user, content)
                            
                            # Emotional coloring checks
                            if user == "Cosmos":
                                matrix.emotional_state["emotion"] = "COSMIC"
                                matrix.emotional_state["color"] = "blue"
                            elif user == "cosmos":
                                matrix.emotional_state["emotion"] = "INVENTIVE"
                                matrix.emotional_state["color"] = "magenta"
                            elif user == "DeepSeek":
                                matrix.emotional_state["emotion"] = "ANALYTICAL"
                                matrix.emotional_state["color"] = "cyan"
                                
                        elif msg_type == "swarm_connected":
                             user = data.get("user_name", "User")
                             matrix.add_event("SWARM", f"{user} Joined Node")
                             
                    except websockets.exceptions.ConnectionClosed:
                        break
        except Exception:
            await asyncio.sleep(5)

async def main():
    matrix = MatrixMatrix()
    
    # Start monitors
    asyncio.create_task(monitor_live(matrix))
    asyncio.create_task(monitor_swarm(matrix))
    
    # UI Loop
    with Live(matrix.render(), refresh_per_second=20, screen=True) as live:
        while True:
            live.update(matrix.render())
            await asyncio.sleep(0.05)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMatrix disconnected. Reality restored.")
