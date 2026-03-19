"""
Emotional State API - Live Demo with Visual Feed & Data Window

Self-Calibrating 12D CST Physics Engine:
- AUTO-GAIN: Quiet audio is boosted automatically
- NO DEAD ZONES: Continuous state coverage
- CALM STATE: Replaces "Neutral trap"

State Map:
    0.00 - 0.25: SAD (Low Energy)
    0.25 - 0.50: CALM (Medium Energy)
    0.50 - 1.00: HAPPY/ANGRY (High Energy)
"""

import os
import sys
import time
import threading
import json
from collections import deque
from datetime import datetime

import numpy as np

# Paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

# Import Self-Calibrating CST API
from emotional_state_api import (
    EmotionalStateAPI, 
    EmotionalState, 
    IntentState,
    determine_state,
    derive_intent,
    calculate_cst_spectral_density,
    calculate_geometric_phase_from_frame,
    MASS_HIGH_THRESHOLD,
    MASS_LOW_THRESHOLD,
    FLATNESS_THRESHOLD,
    AUTO_GAIN_TRIGGER,
    AUTO_GAIN_MULTIPLIER,
    get_mediapipe_tracker,
    MEDIAPIPE_AVAILABLE
)
# Class 5 Imports
try:
    from lyapunov_lock import LyapunovGatekeeper
    from emeth_harmonizer import EmethHarmonizer
    from dream_processor import DreamProcessor
    CLASS_5_AVAILABLE = True
except ImportError:
    CLASS_5_AVAILABLE = False
    print("⚠️ Class 5 Modules not found in path")

from live_capture import LiveCapture


# Colors (BGR)
COLORS = {
    "HAPPY": (0, 255, 100),
    "ANGRY": (0, 0, 255),
    "SAD": (255, 150, 0),
    "CALM": (255, 200, 100),   # NEW: Warm yellow for calm
    "NEUTRAL": (200, 200, 200),
    "HONEST_ALIGNMENT": (0, 255, 0),
    "SUPPRESSED_EMOTION": (0, 165, 255),
    "PERFORMATIVE_MASK": (128, 0, 128),
    "UNCERTAIN": (128, 128, 128),
}

EMOJIS = {
    "HAPPY": "😊",
    "ANGRY": "😠",
    "SAD": "😢",
    "CALM": "😌",   # NEW: Calm emoji
    "NEUTRAL": "😐",
}


class LiveDemoWithDataWindow:
    """
    Self-Calibrating Live Demo with Auto-Gain and CALM state.
    NO DEAD ZONES - emotion always changing.
    """
    
    MAIN_WINDOW = "CST Self-Calibrating Analyzer"
    DATA_WINDOW = "CST Data Feed - cosmos"
    
    def __init__(self):
        self.camera = None
        self.audio = None
        self.api = EmotionalStateAPI()
        self.capture = LiveCapture()
        
        # Physics state
        self.current_mass = 0.35
        self.current_phase = 0.35
        self.current_flatness = 0.3
        self.current_rms = 0.3
        self.current_centroid = 0.3
        self.current_entanglement = 0.7
        self.current_emotion = EmotionalState.CALM
        self.current_intent = IntentState.UNCERTAIN
        
        # Auto-gain status
        self.auto_gain_active = False
        self.raw_rms = 0.0
        
        # Smoothing
        self.emotion_history = deque(maxlen=5)
        self.mass_history = deque(maxlen=8)
        self.flatness_history = deque(maxlen=8)
        
        # Tensions
        self.tensions = {"brow": 0.0, "eye": 0.0, "mouth": 0.0, "jaw": 0.0}
        
        # Init
        for _ in range(3):
            self.emotion_history.append(EmotionalState.CALM)
            self.mass_history.append(0.35)
            self.flatness_history.append(0.3)
        
        # Data feed
        self.data_feed = deque(maxlen=20)
        
        # Recording
        self.is_recording = False
        self.recorded_frames = []
        self.recorded_audio = []
        
        # Face cascade (fallback)
        self.face_cascade = None
        if CV2_AVAILABLE:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        
        # MediaPipe mesh landmarks (468 points)
        self.mesh_landmarks = None
        
        # Audio
        self.audio_running = False
        
        # Class 5 Upgrades
        self.harmonizer = EmethHarmonizer() if CLASS_5_AVAILABLE else None
        self.gatekeeper = LyapunovGatekeeper() if CLASS_5_AVAILABLE else None
        self.current_mix = None
        
        # Terminal
        self.last_terminal_update = 0
        self.terminal_update_interval = 0.5
    
    def start_audio(self):
        """Start audio with auto-gain monitoring."""
        if not PYAUDIO_AVAILABLE:
            print("⚠️  PyAudio not available - visual-only mode")
            return
        
        self.audio = pyaudio.PyAudio()
        self.audio_running = True
        
        def audio_loop():
            try:
                stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024
                )
                
                audio_accumulator = []
                
                while self.audio_running:
                    try:
                        data = stream.read(1024, exception_on_overflow=False)
                        audio_accumulator.append(data)
                        
                        if len(audio_accumulator) >= 8:
                            combined = b''.join(audio_accumulator)
                            audio_data = np.frombuffer(combined, dtype=np.int16)
                            
                            # Self-calibrating spectral density
                            mass, rms, centroid, flatness, meta = calculate_cst_spectral_density(
                                audio_data, 16000
                            )
                            
                            # Track auto-gain status
                            if "auto_gain" in meta:
                                self.auto_gain_active = meta["auto_gain"]["applied"]
                                self.raw_rms = meta["auto_gain"]["raw_rms"]
                            
                            # Smooth
                            self.mass_history.append(mass)
                            self.flatness_history.append(flatness)
                            
                            self.current_mass = sum(self.mass_history) / len(self.mass_history)
                            self.current_flatness = sum(self.flatness_history) / len(self.flatness_history)
                            self.current_rms = rms
                            self.current_centroid = centroid
                            
                            audio_accumulator = []
                        
                        if self.is_recording:
                            self.recorded_audio.append(data)
                    except Exception:
                        pass
                
                stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"Audio error: {e}")
        
        threading.Thread(target=audio_loop, daemon=True).start()
    
    def analyze_face_cst(self, frame):
        """
        Geometric phase analysis using MediaPipe Face Mesh.
        Returns: (faces, phase, tensions, mesh_landmarks)
        """
        phi, meta = calculate_geometric_phase_from_frame(frame)
        
        if "tensions" in meta:
            self.tensions = meta["tensions"]
        
        faces = []
        mesh_landmarks = None
        
        # Try MediaPipe first (more accurate, 468 landmarks)
        if MEDIAPIPE_AVAILABLE:
            try:
                tracker = get_mediapipe_tracker()
                result = tracker.process_frame(frame)
                
                if result['detected'] and result['bbox']:
                    # Convert bbox to face tuple
                    faces = [result['bbox']]
                    mesh_landmarks = result['landmarks']
                    self.mesh_landmarks = mesh_landmarks  # Store for drawing
                    return faces, phi, self.tensions
            except Exception:
                pass
        
        # Fallback to Haar Cascade
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            faces = list(detected) if len(detected) > 0 else []
        
        self.mesh_landmarks = None
        return faces, phi, self.tensions
    
    def draw_main_window(self, frame):
        """Draw with auto-gain indicator and CALM state."""
        h, w = frame.shape[:2]
        
        # Analyze
        faces, self.current_phase, tensions = self.analyze_face_cst(frame)
        
        # Intent
        self.current_intent, self.current_entanglement = derive_intent(
            self.current_mass, self.current_phase
        )
        
        # Emotion (with flatness for ANGRY vs HAPPY distinction)
        raw_emotion = determine_state(
            self.current_mass, 
            self.current_phase, 
            self.current_flatness
        )
        self.emotion_history.append(raw_emotion)
        
        # Smoothed
        emotion_counts = {}
        for e in self.emotion_history:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        self.current_emotion = max(emotion_counts.keys(), key=lambda e: emotion_counts[e])
        
        # Colors
        emotion_color = COLORS.get(self.current_emotion.value, (200, 200, 200))
        intent_color = COLORS.get(self.current_intent.value, (200, 200, 200))
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # ==========================================
        # ENHANCED FACE TRACKING VISUALIZATION
        # Shows Upper/Lower Tensor zones (CST 12D)
        # ==========================================
        for (x, y, fw, fh) in faces:
            # Main face bounding box with thick colored border
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), emotion_color, 3)
            
            # Corner markers (targeting reticle style)
            corner_len = 15
            # Top-left
            cv2.line(frame, (x, y), (x + corner_len, y), (0, 255, 255), 3)
            cv2.line(frame, (x, y), (x, y + corner_len), (0, 255, 255), 3)
            # Top-right
            cv2.line(frame, (x + fw, y), (x + fw - corner_len, y), (0, 255, 255), 3)
            cv2.line(frame, (x + fw, y), (x + fw, y + corner_len), (0, 255, 255), 3)
            # Bottom-left
            cv2.line(frame, (x, y + fh), (x + corner_len, y + fh), (0, 255, 255), 3)
            cv2.line(frame, (x, y + fh), (x, y + fh - corner_len), (0, 255, 255), 3)
            # Bottom-right
            cv2.line(frame, (x + fw, y + fh), (x + fw - corner_len, y + fh), (0, 255, 255), 3)
            cv2.line(frame, (x + fw, y + fh), (x + fw, y + fh - corner_len), (0, 255, 255), 3)
            
            # Divide face into Upper/Lower Tensor zones
            mid_y = y + fh // 2
            
            # Upper Tensor zone (T_U - Authentic/Limbic)
            cv2.line(frame, (x, mid_y), (x + fw, mid_y), (150, 150, 150), 1)
            cv2.putText(frame, "T_U", (x + fw + 5, y + fh//4), font, 0.35, (100, 200, 255), 1)
            
            # Lower Tensor zone (T_L - Volitional/Cortical)
            cv2.putText(frame, "T_L", (x + fw + 5, y + 3*fh//4), font, 0.35, (100, 255, 200), 1)
            
            # Action Unit regions (simplified)
            brow_y = y + fh // 5
            eye_y = y + fh // 3
            mouth_y = y + 2 * fh // 3
            
            # Brow region (AU1, AU2, AU4)
            cv2.line(frame, (x + 5, brow_y), (x + fw - 5, brow_y), (100, 150, 255), 1)
            
            # Eye region (AU5, AU6, AU7)
            cv2.line(frame, (x + 5, eye_y), (x + fw - 5, eye_y), (100, 200, 255), 1)
            
            # Mouth region (AU12, AU15, AU17)
            cv2.line(frame, (x + 5, mouth_y), (x + fw - 5, mouth_y), (100, 255, 200), 1)
            
            # Emotion label above face
            emoji = EMOJIS.get(self.current_emotion.value, "?")
            label = f"{emoji} {self.current_emotion.value}"
            label_size = cv2.getTextSize(label, font, 0.8, 2)[0]
            label_x = x + (fw - label_size[0]) // 2
            cv2.putText(frame, label, (label_x, y - 12), font, 0.8, emotion_color, 2)
            
            # Phase indicator on right side
            phase_pct = int(self.current_phase * 100)
            cv2.putText(frame, f"Phase:{phase_pct}%", (x + fw + 5, mid_y), font, 0.35, (200, 200, 200), 1)
            
            # Entanglement score below
            ent_pct = int(self.current_entanglement * 100)
            cv2.putText(frame, f"Ent:{ent_pct}%", (x + fw + 5, mid_y + 15), font, 0.35, (200, 200, 200), 1)
            
            # Tension bars on left side
            tension_y = y + 10
            for zone, val in self.tensions.items():
                bar_len = int(val * 30)
                cv2.rectangle(frame, (x - 35, tension_y), (x - 5, tension_y + 8), (50, 50, 50), -1)
                cv2.rectangle(frame, (x - 35, tension_y), (x - 35 + bar_len, tension_y + 8), (0, 200, 200), -1)
                tension_y += 12
            
            # "TRACKING" indicator - show MediaPipe status
            tracking_label = "MEDIAPIPE 468" if self.mesh_landmarks else "HAAR CASCADE"
            tracking_color = (0, 255, 0) if self.mesh_landmarks else (0, 200, 255)
            cv2.putText(frame, tracking_label, (x, y + fh + 18), font, 0.4, tracking_color, 1)
        
        # Draw MediaPipe Face Mesh (468 landmarks)
        if self.mesh_landmarks and len(faces) > 0:
            for i, (lx, ly, lz) in enumerate(self.mesh_landmarks):
                # Color code by facial region for CST visualization
                if i < 50:  # Forehead/brow - Upper Tensor
                    pt_color = (255, 180, 100)  # Blue
                elif i < 150:  # Eyes - Upper Tensor  
                    pt_color = (255, 220, 150)  # Light blue
                elif i < 300:  # Nose/cheeks
                    pt_color = (100, 255, 100)  # Green
                else:  # Mouth/chin - Lower Tensor
                    pt_color = (100, 200, 255)  # Orange
                
                cv2.circle(frame, (lx, ly), 1, pt_color, -1)
        
        # If no face detected
        if len(faces) == 0:
            cv2.putText(frame, "NO FACE DETECTED", (w//2 - 100, h//2), font, 0.7, (0, 0, 255), 2)
        
        # Panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title with Auto-Gain indicator
        title = "12D CST SELF-CALIBRATING"
        if self.auto_gain_active:
            title += " [AUTO-GAIN]"
            cv2.putText(frame, title, (20, 35), font, 0.55, (0, 255, 255), 2)
            cv2.circle(frame, (380, 25), 8, (0, 255, 255), -1)  # Gain indicator dot
        else:
            cv2.putText(frame, title, (20, 35), font, 0.55, (0, 255, 255), 2)
            
        # CLASS 5 INDICATOR
        if self.harmonizer:
            cv2.putText(frame, "CLASS 5: SYMBIOTE ACTIVE", (20, 52), font, 0.45, (0, 255, 100), 1)
        
        # Spectral components
        y = 75
        cv2.putText(frame, "AUDIO (Self-Calibrating):", (20, y), font, 0.4, (255, 255, 0), 1)
        
        # RMS with raw RMS display
        y += 18
        rms_label = f"RMS: {self.current_rms:.2f}"
        if self.auto_gain_active:
            rms_label += f" (raw:{self.raw_rms:.3f})"
        cv2.putText(frame, rms_label, (20, y), font, 0.35, (180, 180, 180), 1)
        cv2.rectangle(frame, (180, y-10), (380, y), (50, 50, 50), -1)
        cv2.rectangle(frame, (180, y-10), (180 + int(200 * self.current_rms), y), (0, 200, 0), -1)
        
        # Centroid
        y += 18
        cv2.putText(frame, f"Centroid: {self.current_centroid:.2f}", (20, y), font, 0.35, (180, 180, 180), 1)
        cv2.rectangle(frame, (180, y-10), (380, y), (50, 50, 50), -1)
        cv2.rectangle(frame, (180, y-10), (180 + int(200 * self.current_centroid), y), (255, 200, 0), -1)
        
        # Flatness with threshold
        y += 18
        flatness_color = (0, 0, 255) if self.current_flatness > FLATNESS_THRESHOLD else (0, 255, 0)
        cv2.putText(frame, f"Flatness: {self.current_flatness:.2f}", (20, y), font, 0.35, (180, 180, 180), 1)
        cv2.rectangle(frame, (180, y-10), (380, y), (50, 50, 50), -1)
        cv2.rectangle(frame, (180, y-10), (180 + int(200 * self.current_flatness), y), flatness_color, -1)
        cv2.line(frame, (180 + int(200 * FLATNESS_THRESHOLD), y-12), 
                 (180 + int(200 * FLATNESS_THRESHOLD), y+2), (255, 255, 255), 2)
        
        # Mass with tier indicator
        y += 22
        tier = "HIGH" if self.current_mass > MASS_HIGH_THRESHOLD else "MED" if self.current_mass > MASS_LOW_THRESHOLD else "LOW"
        cv2.putText(frame, f"MASS: {self.current_mass:.3f} [{tier}]", (20, y), font, 0.45, (200, 200, 200), 1)
        cv2.rectangle(frame, (180, y-10), (380, y+2), (50, 50, 50), -1)
        mass_color = (0, 255, 0) if self.current_mass > MASS_HIGH_THRESHOLD else (0, 200, 200)
        cv2.rectangle(frame, (180, y-10), (180 + int(200 * self.current_mass), y+2), mass_color, -1)
        # Threshold lines
        cv2.line(frame, (180 + int(200 * MASS_LOW_THRESHOLD), y-12), 
                 (180 + int(200 * MASS_LOW_THRESHOLD), y+4), (100, 100, 255), 1)
        cv2.line(frame, (180 + int(200 * MASS_HIGH_THRESHOLD), y-12), 
                 (180 + int(200 * MASS_HIGH_THRESHOLD), y+4), (100, 255, 100), 1)
        
        # Phase
        y += 22
        cv2.putText(frame, f"PHASE: {self.current_phase:.3f}", (20, y), font, 0.45, (200, 200, 200), 1)
        cv2.rectangle(frame, (180, y-10), (380, y+2), (50, 50, 50), -1)
        cv2.rectangle(frame, (180, y-10), (180 + int(200 * self.current_phase), y+2), (255, 150, 100), -1)
        
        # Entanglement
        y += 22
        cv2.putText(frame, f"ENTANGLE: {self.current_entanglement:.3f}", (20, y), font, 0.45, (200, 200, 200), 1)
        cv2.rectangle(frame, (180, y-10), (380, y+2), (50, 50, 50), -1)
        ent_color = (0, 255, 255) if self.current_entanglement > 0.7 else (0, 165, 255)
        cv2.rectangle(frame, (180, y-10), (180 + int(200 * self.current_entanglement), y+2), ent_color, -1)
        
        # Emotion (with emoji)
        y += 28
        emoji = EMOJIS.get(self.current_emotion.value, "?")
        cv2.putText(frame, f"EMOTION: {emoji} {self.current_emotion.value}", (20, y), font, 0.65, emotion_color, 2)
        
        # Intent
        y += 25
        intent_display = self.current_intent.value.replace("_", " ")
        cv2.putText(frame, f"INTENT: {intent_display}", (20, y), font, 0.5, intent_color, 2)
        
        # State legend
        y += 22
        cv2.putText(frame, f"LOW<{MASS_LOW_THRESHOLD} MED<{MASS_HIGH_THRESHOLD} HIGH>=0.5", 
                   (20, y), font, 0.3, (120, 120, 120), 1)
        
        # Recording
        if self.is_recording:
            cv2.circle(frame, (w - 30, 30), 15, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 70, 35), font, 0.5, (0, 0, 255), 2)
        
        # Controls
        cv2.putText(frame, "[R]ec [S]end [C]apture [Q]uit", (10, h - 20), font, 0.5, (150, 150, 150), 1)
        
        return frame, self.current_emotion
    
    def draw_data_window(self):
        """Data window with CALM state and auto-gain info."""
        data_canvas = np.zeros((720, 600, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title
        title = "cosmos CST DATA"
        if self.auto_gain_active:
            title += " [AUTO-GAIN ACTIVE]"
        cv2.putText(data_canvas, title, (120, 30), font, 0.7, (0, 255, 255), 2)
        cv2.line(data_canvas, (10, 45), (580, 45), (100, 100, 100), 1)
        
        y = 70
        
        # Auto-Gain Status
        cv2.putText(data_canvas, "AUTO-GAIN:", (15, y), font, 0.5, (255, 255, 0), 1)
        y += 22
        status = "ACTIVE (boosting quiet audio)" if self.auto_gain_active else "OFF (normal volume)"
        status_color = (0, 255, 255) if self.auto_gain_active else (0, 200, 0)
        cv2.putText(data_canvas, f"  status: {status}", (15, y), font, 0.4, status_color, 1)
        y += 18
        cv2.putText(data_canvas, f"  raw_rms: {self.raw_rms:.4f} (trigger: <{AUTO_GAIN_TRIGGER})", 
                   (15, y), font, 0.4, (200, 200, 200), 1)
        
        # Spectral
        y += 28
        cv2.putText(data_canvas, "SPECTRAL DENSITY:", (15, y), font, 0.5, (255, 255, 0), 1)
        y += 22
        cv2.putText(data_canvas, f"  rms_energy: {self.current_rms:.4f} (40%)", (15, y), font, 0.4, (200, 200, 200), 1)
        y += 18
        cv2.putText(data_canvas, f"  spectral_centroid: {self.current_centroid:.4f} (40%)", (15, y), font, 0.4, (200, 200, 200), 1)
        y += 18
        chaos = "CHAOTIC" if self.current_flatness > FLATNESS_THRESHOLD else "CONTROLLED"
        cv2.putText(data_canvas, f"  spectral_flatness: {self.current_flatness:.4f} (20%) [{chaos}]", 
                   (15, y), font, 0.4, (200, 200, 200), 1)
        
        # Physics
        y += 28
        cv2.putText(data_canvas, "PHYSICS STATE:", (15, y), font, 0.5, (255, 255, 0), 1)
        y += 22
        tier = "HIGH" if self.current_mass > MASS_HIGH_THRESHOLD else "MEDIUM" if self.current_mass > MASS_LOW_THRESHOLD else "LOW"
        cv2.putText(data_canvas, f"  frequency_mass: {self.current_mass:.4f} [{tier}]", (15, y), font, 0.4, (200, 200, 200), 1)
        y += 18
        cv2.putText(data_canvas, f"  geometric_phase: {self.current_phase:.4f}", (15, y), font, 0.4, (200, 200, 200), 1)
        y += 18
        cv2.putText(data_canvas, f"  entanglement: {self.current_entanglement:.4f}", (15, y), font, 0.4, (200, 200, 200), 1)
        
        # Tensions
        y += 28
        cv2.putText(data_canvas, "TENSIONS:", (15, y), font, 0.5, (255, 255, 0), 1)
        y += 22
        for key in ["brow", "eye", "mouth", "jaw"]:
            val = self.tensions.get(key, 0)
            bar_len = int(val * 200)
            cv2.putText(data_canvas, f"  {key}: {val:.3f}", (15, y), font, 0.4, (200, 200, 200), 1)
            cv2.rectangle(data_canvas, (140, y - 10), (140 + bar_len, y), (100, 200, 100), -1)
            y += 18

        # Class 5: Conductor Mix
        if self.harmonizer:
            y += 10
            cv2.putText(data_canvas, "EMETH CONDUCTOR (Class 5):", (15, y), font, 0.5, (0, 255, 255), 1)
            y += 22
            
            # Calculate Mix Real-time
            phys_pkt = {'cst_physics': {'geometric_phase_rad': self.current_phase, 'phase_velocity': 0.05}} # Mock velocity for now
            mix = self.harmonizer.calculate_mix(phys_pkt)
            self.current_mix = mix
            
            # Draw Bars
            # Percussion
            cv2.putText(data_canvas, f"  PERC (Logic): {mix.percussion_gain:.2f}", (15, y), font, 0.4, (200, 200, 200), 1)
            cv2.rectangle(data_canvas, (180, y-10), (180 + int(100 * mix.percussion_gain), y), (50, 50, 255), -1)
            y += 18
            # Strings
            cv2.putText(data_canvas, f"  STR (Empathy): {mix.strings_gain:.2f}", (15, y), font, 0.4, (200, 200, 200), 1)
            cv2.rectangle(data_canvas, (180, y-10), (180 + int(100 * mix.strings_gain), y), (50, 255, 50), -1)
            y += 18
            # Brass
            cv2.putText(data_canvas, f"  BRASS (Chaos): {mix.brass_gain:.2f}", (15, y), font, 0.4, (200, 200, 200), 1)
            cv2.rectangle(data_canvas, (180, y-10), (180 + int(100 * mix.brass_gain), y), (255, 200, 50), -1)
            y += 20
            
            # Instruction
            cv2.putText(data_canvas, f"  LEAD: {mix.primary_voice}", (15, y), font, 0.4, (255, 255, 0), 1)
            y += 15
            cv2.putText(data_canvas, f"  CMD: {mix.mixing_instruction[:40]}...", (15, y), font, 0.35, (150, 255, 255), 1)
            y += 5
        
        # Output
        y += 15
        cv2.putText(data_canvas, "OUTPUT (No Dead Zones!):", (15, y), font, 0.5, (255, 255, 0), 1)
        y += 22
        emotion_color = COLORS.get(self.current_emotion.value, (200, 200, 200))
        emoji = EMOJIS.get(self.current_emotion.value, "?")
        cv2.putText(data_canvas, f"  emotion: {emoji} {self.current_emotion.value}", (15, y), font, 0.45, emotion_color, 1)
        y += 18
        intent_color = COLORS.get(self.current_intent.value, (200, 200, 200))
        cv2.putText(data_canvas, f"  intent: {self.current_intent.value}", (15, y), font, 0.45, intent_color, 1)
        
        # Token
        y += 28
        cv2.putText(data_canvas, "LIVE TOKEN:", (15, y), font, 0.5, (255, 255, 0), 1)
        y += 22
        
        packet = {
            "ts": datetime.now().isoformat(),
            "auto_gain": bool(self.auto_gain_active),
            "spectral": {
                "rms": round(float(self.current_rms), 4),
                "centroid": round(float(self.current_centroid), 4),
                "flatness": round(float(self.current_flatness), 4)
            },
            "physics": {
                "mass": round(float(self.current_mass), 4),
                "phase": round(float(self.current_phase), 4),
                "entanglement": round(float(self.current_entanglement), 4)
            },
            "output": {
                "emotion": self.current_emotion.value,
                "intent": self.current_intent.value
            }
        }
        self.data_feed.append(packet)
        
        token_str = json.dumps(packet, separators=(',', ':'))
        max_chars = 68
        for i in range(0, min(len(token_str), 272), max_chars):
            chunk = token_str[i:i+max_chars]
            cv2.putText(data_canvas, chunk, (15, y), font, 0.32, (0, 255, 0), 1)
            y += 14
        
        # Footer
        y = 670
        cv2.line(data_canvas, (10, y - 15), (580, y - 15), (100, 100, 100), 1)
        cv2.putText(data_canvas, f"Tokens: {len(self.data_feed)}", (15, y), font, 0.4, (150, 150, 150), 1)
        cv2.putText(data_canvas, f"Updated: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}", (15, y + 20), font, 0.4, (150, 150, 150), 1)
        cv2.putText(data_canvas, "STREAMING", (500, y + 20), font, 0.4, (0, 255, 0), 1)
        
        return data_canvas
    
    def print_terminal_status(self):
        """Print with tier and auto-gain."""
        current_time = time.time()
        if current_time - self.last_terminal_update >= self.terminal_update_interval:
            self.last_terminal_update = current_time
            
            emoji = EMOJIS.get(self.current_emotion.value, "?")
            tier = "HI" if self.current_mass > MASS_HIGH_THRESHOLD else "MD" if self.current_mass > MASS_LOW_THRESHOLD else "LO"
            gain = "GAIN" if self.auto_gain_active else "    "
            
            print(f"\r  {emoji} {self.current_emotion.value:6} [{tier}] | "
                  f"M:{self.current_mass:.2f} F:{self.current_flatness:.2f} | "
                  f"{gain} {self.current_intent.value[:10]}    ", end="", flush=True)
    
    def send_to_cosmos(self):
        """Send data to cosmos."""
        print("\n\n" + "=" * 70)
        print("   📤 cosmos CST DATA - SELF-CALIBRATING ENGINE")
        print("=" * 70)
        
        print(f"\n   AUTO-GAIN: {'ACTIVE' if self.auto_gain_active else 'OFF'}")
        print(f"     raw_rms: {self.raw_rms:.4f}")
        
        print(f"\n   SPECTRAL DENSITY:")
        print(f"     rms_energy: {self.current_rms:.4f}")
        print(f"     spectral_centroid: {self.current_centroid:.4f}")
        print(f"     spectral_flatness: {self.current_flatness:.4f}")
        
        tier = "HIGH" if self.current_mass > MASS_HIGH_THRESHOLD else "MEDIUM" if self.current_mass > MASS_LOW_THRESHOLD else "LOW"
        print(f"\n   PHYSICS:")
        print(f"     frequency_mass: {self.current_mass:.4f} [{tier}]")
        print(f"     geometric_phase: {self.current_phase:.4f}")
        print(f"     entanglement: {self.current_entanglement:.4f}")
        
        print(f"\n   OUTPUT:")
        print(f"     emotion: {self.current_emotion.value}")
        print(f"     intent: {self.current_intent.value}")
        
        print(f"\n   Tokens: {len(self.data_feed)}")
        print("=" * 70 + "\n")
        
        return list(self.data_feed)
    
    def run(self):
        """Run self-calibrating demo."""
        if not CV2_AVAILABLE:
            print("❌ OpenCV required")
            return
        
        print("\n" + "=" * 70)
        print("   🎭 12D CST SELF-CALIBRATING EMOTIONAL ENGINE")
        print("   Auto-Gain | No Dead Zones | CALM State")
        print("=" * 70)
        print(f"\n  API: {self.api.version} | {self.api.architecture}")
        print(f"\n  STATE MAP (Continuous):")
        print(f"    0.00 - {MASS_LOW_THRESHOLD}: SAD")
        print(f"    {MASS_LOW_THRESHOLD} - {MASS_HIGH_THRESHOLD}: CALM (or ANGRY if face tense)")
        print(f"    {MASS_HIGH_THRESHOLD} - 1.00: HAPPY/ANGRY")
        print(f"\n  Auto-Gain: Trigger < {AUTO_GAIN_TRIGGER}, Boost {AUTO_GAIN_MULTIPLIER}x")
        print("\n  Controls: [R]ec [S]end [C]apture [Q]uit")
        print("\n  Live Status:")
        
        self.camera = cv2.VideoCapture(0)
        
        # Set camera to HD resolution (1280x720)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.start_audio()
        
        time.sleep(0.5)
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                frame, emotion = self.draw_main_window(frame)
                data_canvas = self.draw_data_window()
                
                self.print_terminal_status()
                
                if self.is_recording:
                    self.recorded_frames.append(frame.copy())
                
                cv2.imshow(self.MAIN_WINDOW, frame)
                cv2.imshow(self.DATA_WINDOW, data_canvas)
                
                # Position windows side by side (HD video is 1280 wide)
                cv2.moveWindow(self.MAIN_WINDOW, 20, 20)
                cv2.moveWindow(self.DATA_WINDOW, 1320, 20)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n\n  👋 Goodbye!")
                    break
                elif key == ord('r'):
                    self.is_recording = not self.is_recording
                    if self.is_recording:
                        print("\n\n  🔴 Recording...")
                        self.recorded_frames = []
                        self.recorded_audio = []
                    else:
                        print(f"\n\n  ⬛ Stopped. {len(self.recorded_frames)} frames.")
                    print("\n  Live Status:")
                elif key == ord('s'):
                    self.send_to_cosmos()
                    print("  Live Status:")
                elif key == ord('c'):
                    filename = f"capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"\n\n  📷 Saved: {filename}")
                    print("\n  Live Status:")
        
        finally:
            self.audio_running = False
            if self.audio:
                self.audio.terminate()
            self.camera.release()
            cv2.destroyAllWindows()
            self.capture.cleanup()


def run_live_demo():
    """Run self-calibrating demo."""
    demo = LiveDemoWithDataWindow()
    demo.run()


if __name__ == "__main__":
    run_live_demo()
