"""
cosmos Full System - 12D CST Complete Integration

Runs the complete emotional intelligence system:
1. MediaPipe Face Mesh (468 landmarks)
2. Real-time Action Unit detection
3. Emotional token server (port 8765)
4. cosmos AI companion integration

Usage:
    python full_system.py
    python full_system.py --no-display  # Headless mode
"""

import os
import sys
import cv2
import time
import json
import asyncio
import threading
import argparse
from collections import deque
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import CST components
from emotional_state_api import (
    EmotionalStateAPI,
    EmotionalState,
    IntentState,
    CSTPhaseState,
    LLMPersonaMode,
    get_mediapipe_tracker,
    estimate_action_units_from_frame,
    calculate_geometric_phase,
    MEDIAPIPE_AVAILABLE,
    MEDIAPIPE_AVAILABLE,
    PHASE_SYNCHRONY
)

from dream_processor import DreamProcessor

# Try FastAPI
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️  FastAPI not installed. Run: pip install fastapi uvicorn")


class FullSystemController:
    """
    Full system controller that integrates:
    - Camera capture
    - MediaPipe face tracking  
    - Real-time emotional state API
    - Token server for cosmos AI
    """
    
    def __init__(self):
        self.api = EmotionalStateAPI()
        self.tracker = get_mediapipe_tracker() if MEDIAPIPE_AVAILABLE else None
        
        # Camera
        self.camera = None
        self.current_frame = None
        self.camera_running = False
        
        # State
        self.current_packet = None
        self.face_detected = False
        self.landmarks = None
        self.fps = 0
        self.frame_count = 0
        
        # Display
        self.show_display = True
        
        # Server
        self.server_task = None
        self.connected_clients = []

        # Class 5: Dream Processor
        self.dreamer = DreamProcessor()
        self.interaction_log = [] # Short-term memory buffer
        self.last_face_time = time.time()
        self.is_dreaming = False
        self.min_mass_threshold = 20.0 # Only log events > 20 mass
        
        # Audio Pipe (Swarm Acoustic Input)
        try:
            # Ensure project root is on the path (emotional_api is a sibling of Cosmos/)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from Cosmos.core.multimodal.real_time_audio_pipe import RealTimeAudioPipe
            self.audio_pipe = RealTimeAudioPipe()
            self.audio_pipe.start()
            print("🎤 Real-Time Audio Pipeline Started")
        except Exception as e:
            print(f"⚠️ Audio Pipeline Failed: {e}")
            self.audio_pipe = None
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║         🧠 cosmos FULL SYSTEM - 12D CST INTEGRATION         ║
╠══════════════════════════════════════════════════════════════════╣
║  MediaPipe:  {'✅ 468 Landmarks' if MEDIAPIPE_AVAILABLE else '⚠️  Haar Cascade Fallback'}  
║  FastAPI:    {'✅ Available' if FASTAPI_AVAILABLE else '❌ Not Installed'}
║  API:        v{self.api.version}         
╚══════════════════════════════════════════════════════════════════╝
        """)
    
    def process_frame(self, frame):
        """Process a camera frame and update emotional state."""
        if frame is None:
            return None
        
        # Get face landmarks and Action Units
        if self.tracker:
            result = self.tracker.process_frame(frame)
            self.face_detected = result['detected']
            self.landmarks = result['landmarks']
            
            if result['detected'] and result['blendshapes']:
                upper, lower = self.tracker.get_action_units(result['blendshapes'])
                
                # Calculate geometric phase from real AU data
                phase = calculate_geometric_phase(upper, lower)
                
                # Update API with real data
                self.api.update_from_tensors(upper, lower, phase)
                
                # Reset Sleep Timer
                self.last_face_time = time.time()
                if self.is_dreaming:
                    print("👀 USER DETECTED - WAKING FROM DREAM STATE")
                    self.is_dreaming = False
                    
        else:
            # Fallback estimation
            upper, lower = estimate_action_units_from_frame(frame)
            phase = calculate_geometric_phase(upper, lower)
            self.api.update_from_tensors(upper, lower, phase)
            self.face_detected = True # Assume detected in fallback for now
            
        # Check Sleep Condition (No face for > X time)
        time_since_face = time.time() - self.last_face_time
        if time_since_face > 30 and not self.is_dreaming:
             # Check Entropy (Virtual Body taking over)
             entropy = self.api.virtual_body.entropy
             if entropy > 0.9 and len(self.interaction_log) > 0:
                 self.is_dreaming = True
                 print(f"💤 SYSTEM ENTERING REM SLEEP (Entropy: {entropy:.2f})...")
                 
                 # TRIGGER DREAM PROCESSOR
                 self.dreamer.process_daily_logs(self.interaction_log)
                 
                 # Clear Short Term Memory
                 self.interaction_log = []
                 print("☀️ WAKING UP UPDATED.")
                 self.last_face_time = time.time() # Reset to prevent loop
        
        # Get packet data (using stored state)
        self.current_packet = self.api.get_cosmos_packet()
        return self.current_packet
    
    def draw_overlay(self, frame):
        """Draw CST overlay on frame."""
        if frame is None:
            return frame
        
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw face mesh
        if self.landmarks:
            for i, (x, y, z) in enumerate(self.landmarks):
                if i < 100:  # Upper face
                    color = (255, 180, 100)  # Blue
                elif i < 300:  # Mid face
                    color = (100, 255, 100)  # Green
                else:  # Lower face
                    color = (100, 200, 255)  # Orange
                cv2.circle(frame, (x, y), 1, color, -1)
        
        # Status panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "cosmos FULL SYSTEM", (20, 35), font, 0.6, (0, 255, 255), 2)
        
        if self.current_packet:
            physics = self.current_packet.get('cst_physics', {})
            meta = self.current_packet.get('meta_instruction', {})
            
            # CST State
            state = physics.get('cst_state', 'UNKNOWN')
            mode = meta.get('persona_mode', 'UNKNOWN')
            
            state_colors = {
                'SYNCHRONY': (0, 255, 0),
                'MASKING': (0, 255, 255),
                'LEAKAGE': (0, 100, 255),
                'JITTER': (255, 100, 100)
            }
            color = state_colors.get(state, (200, 200, 200))
            
            cv2.putText(frame, f"CST State: {state}", (20, 60), font, 0.5, color, 1)
            cv2.putText(frame, f"LLM Mode: {mode}", (20, 80), font, 0.5, (200, 200, 200), 1)
            
            # Phase bar
            phase = physics.get('geometric_phase_rad', 0)
            phase_pct = min(1.0, phase / 1.57)
            cv2.putText(frame, f"Phase: {phase:.2f} rad", (20, 100), font, 0.4, (200, 200, 200), 1)
            cv2.rectangle(frame, (130, 90), (300, 105), (50, 50, 50), -1)
            cv2.rectangle(frame, (130, 90), (130 + int(170 * phase_pct), 105), color, -1)
            
            # Entanglement
            ent = physics.get('entanglement_score', 0)
            cv2.putText(frame, f"Entanglement: {ent:.2f}", (20, 120), font, 0.4, (200, 200, 200), 1)
            
            # Server status
            cv2.putText(frame, f"API: localhost:8765", (20, 145), font, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, f"Clients: {len(self.connected_clients)}", (180, 145), font, 0.4, (0, 255, 0), 1)
        
        # Face detection indicator
        indicator = "TRACKING" if self.face_detected else "NO FACE"
        indicator_color = (0, 255, 0) if self.face_detected else (0, 0, 255)
        cv2.putText(frame, indicator, (w - 120, 30), font, 0.6, indicator_color, 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.0f}", (w - 80, h - 20), font, 0.4, (150, 150, 150), 1)
        
        # Controls
        cv2.putText(frame, "Press Q to quit", (10, h - 20), font, 0.4, (150, 150, 150), 1)
        
        return frame
    
    def camera_loop(self):
        """Main camera capture and processing loop."""
        # Use DirectShow (CAP_DSHOW) on Windows to prevent deadlock 
        # with concurrent audio pipeline initialization.
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            # Fallback to default if DSHOW fails
            self.camera = cv2.VideoCapture(0)
            
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera_running = True
        
        last_time = time.time()
        
        print("📷 Camera started. Press Q to quit.")
        
        while self.camera_running:
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            self.current_frame = frame
            
            # Process frame
            self.process_frame(frame)
            
            # Calculate FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                now = time.time()
                self.fps = 30 / (now - last_time)
                last_time = now
            
            # Display
            if self.show_display:
                display_frame = self.draw_overlay(frame.copy())
                cv2.imshow("cosmos Full System", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.camera_running = False
                    break
        
        self.camera.release()
        cv2.destroyAllWindows()
        print("📷 Camera stopped.")
    
    def get_current_packet(self):
        """
        Get the current cosmos_packet.
        - If Face Detected: Return Real-World Data (Symbiosis).
        - If No Face: Return Virtual Body Data (Autonomy).
        """
        if self.face_detected and self.current_packet:
            # Inject Virtual Stats into cst_physics (where UI expects them)
            virtual = self.api.virtual_body.tick()
            
            # Ensure cst_physics exists and add virtual_body there
            if 'cst_physics' not in self.current_packet:
                self.current_packet['cst_physics'] = {}
            self.current_packet['cst_physics']['virtual_body'] = virtual
            
            # Also add at root for backwards compatibility
            self.current_packet['virtual_body'] = virtual
            return self.current_packet
            
        # Fallback to Virtual Embodiment (Ghost Mode)
        return self.api.get_virtual_packet()
    
    def run(self, headless=False):
        """Run the full system."""
        self.show_display = not headless
        
        if not FASTAPI_AVAILABLE:
            print("❌ FastAPI required. Run: pip install fastapi uvicorn")
            return
        
        # Create FastAPI app
        app = FastAPI(title="cosmos Full System")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        controller = self
        
        @app.get("/")
        async def root():
            return JSONResponse({
                "server": "cosmos Full System",
                "version": controller.api.version,
                "status": "online",
                "face_detected": controller.face_detected,
                "mediapipe": MEDIAPIPE_AVAILABLE,
                "fps": round(controller.fps, 1)
            })
        
        @app.get("/state")
        async def get_state():
            return JSONResponse(controller.get_current_packet())
        
        @app.get("/stream")
        async def stream():
            async def generate():
                while True:
                    packet = controller.get_current_packet()
                    yield f"data: {json.dumps(packet)}\n\n"
                    await asyncio.sleep(0.2)  # 5 packets/sec
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        @app.get("/system_prompt")
        async def system_prompt():
            from pathlib import Path
            prompt_path = Path(__file__).parent / "system_prompt.txt"
            if prompt_path.exists():
                return PlainTextResponse(prompt_path.read_text(encoding="utf-8"))
            return PlainTextResponse("Not found", status_code=404)
            
        @app.get("/vision")
        async def get_vision():
            """Returns the current raw webcam frame as a contiguous base64 encoded JPEG."""
            if controller.current_frame is None:
                return JSONResponse({"status": "error", "message": "No frame available (Camera off or loading)"}, status_code=503)
                
            try:
                import base64
                # We need to grab a copy of the frame safely
                frame_copy = controller.current_frame.copy()
                
                # Encode straight to JPEG in memory
                success, buffer = cv2.imencode('.jpg', frame_copy)
                if not success:
                    return JSONResponse({"status": "error", "message": "Failed to encode CV2 frame"}, status_code=500)
                    
                # Base64 encode the byte buffer
                b64_str = base64.b64encode(buffer).decode('utf-8')
                return JSONResponse({
                    "status": "success",
                    "format": "jpeg",
                    "width": frame_copy.shape[1],
                    "height": frame_copy.shape[0],
                    "image": b64_str
                })
            except Exception as e:
                print(f"Vision API Error: {e}")
                return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
                
        @app.get("/audio_tokens")
        async def get_audio_tokens():
            """Returns the latest FFT Phase-Harmonic tokens from the raw microphone."""
            if not hasattr(controller, 'audio_pipe') or controller.audio_pipe is None:
                return JSONResponse({"status": "error", "message": "Audio pipeline not active"}, status_code=503)
                
            try:
                # Fetch the latest tokens from the deque (if any exist)
                # Note: The buffer might have multiple events since last poll, we return all of them
                # so the consumer can see the temporal wave
                tokens = controller.audio_pipe.pop_tokens()
                return JSONResponse({
                    "status": "success",
                    "events": tokens,
                    "count": len(tokens)
                })
            except Exception as e:
                print(f"Audio API Error: {e}")
                return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            controller.connected_clients.append(websocket)
            print(f"✅ Client connected. Total: {len(controller.connected_clients)}")
            
            try:
                await websocket.send_json({
                    "type": "connected",
                    "message": "cosmos Full System - Real Face Data",
                    "version": controller.api.version,
                    "mediapipe": MEDIAPIPE_AVAILABLE
                })
                
                while True:
                    try:
                        packet = controller.get_current_packet()
                        await websocket.send_json({
                            "type": "cosmos_packet",
                            "face_detected": controller.face_detected,
                            **packet
                        })
                        await asyncio.sleep(0.2)
                    except WebSocketDisconnect:
                        print(f"❌ Client disconnected (Loop Break).")
                        break # Exit loop
                    except RuntimeError as e:
                        if "close message has been sent" in str(e):
                            print(f"❌ WebSocket Closed (RuntimeError): {e}")
                            break # Exit loop
                        print(f"⚠️ WebSocket Runtime Error: {e}")
                        await asyncio.sleep(1.0)
                    except Exception as e:
                        print(f"⚠️ WebSocket Inner Error: {e}")
                        await asyncio.sleep(1.0)
                    
            except WebSocketDisconnect:
                pass  # Cleanup in finally
            except Exception as e:
                print(f"❌ WebSocket Critical Error: {e}")
            finally:
                if websocket in controller.connected_clients:
                    controller.connected_clients.remove(websocket)
                print(f"🔌 Client cleaned up. Total: {len(controller.connected_clients)}")
        
        # Start server in background thread so OpenCV can own the main thread
        def run_server():
            # In a thread, uvicorn might have issues with signal handlers, so we can disable them if needed,
            # but usually it runs fine or we can configure it to not use signals.
            config = uvicorn.Config(app, host="0.0.0.0", port=8765, log_level="warning")
            server = uvicorn.Server(config)
            # Disable signal handlers as we are in a thread
            server.install_signal_handlers = lambda: None
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(server.serve())
            except Exception as e:
                print(f"Server Thread Warning: {e}")
            
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                   🚀 FULL SYSTEM RUNNING                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  API Server:   http://localhost:8765                             ║
║                                                                  ║
║  Endpoints:                                                      ║
║    GET  /              - Status                                  ║
║    GET  /state         - Current cosmos_packet               ║
║    GET  /stream        - SSE token stream                        ║
║    WS   /ws            - WebSocket for cosmos AI             ║
║    GET  /system_prompt - LLM steering prompt                     ║
║                                                                  ║
║  Camera:  {'Running' if not headless else 'Headless Mode'}                                             ║
║                                                                  ║
║  Connect cosmos AI to ws://localhost:8765/ws                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
        """)
        
        # Run camera loop in main thread (blocks)
        try:
            if not headless:
                self.camera_loop()
            else:
                # Keep main thread alive if headless
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            self.camera_running = False
            print("Shutting down...")


def main():
    parser = argparse.ArgumentParser(description="cosmos Full System")
    parser.add_argument("--no-display", action="store_true", help="Run headless (no GUI)")
    args = parser.parse_args()
    
    controller = FullSystemController()
    controller.run(headless=args.no_display)


if __name__ == "__main__":
    main()
