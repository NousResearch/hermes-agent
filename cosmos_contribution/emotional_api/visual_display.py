"""
Emotional State API - Visual Display Module

Real-time camera feed with CST emotional analysis overlay.
Shows the user's face with emotion detection visualization.
"""

import os
import sys
import time
import threading
import tempfile
from collections import deque
from typing import Optional, Tuple

import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

from emotional_state_api import EmotionalStateAPI, determine_state, EmotionalState
from live_capture import LiveCapture


# =============================================================================
# VISUAL DISPLAY CONFIGURATION
# =============================================================================

# Window settings
WINDOW_NAME = "🎭 CST Emotional State Analyzer"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Colors (BGR format for OpenCV)
COLORS = {
    "HAPPY": (0, 255, 100),    # Green
    "ANGRY": (0, 0, 255),      # Red
    "SAD": (255, 150, 0),      # Blue
    "NEUTRAL": (200, 200, 200) # Gray
}

# Overlay settings
FONT = cv2.FONT_HERSHEY_SIMPLEX if CV2_AVAILABLE else None
FONT_SCALE = 0.8
FONT_THICKNESS = 2


# =============================================================================
# AUDIO ANALYZER (Background Thread)
# =============================================================================

class AudioAnalyzer:
    """Background audio analysis for continuous emotional mass calculation."""
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.running = False
        self.thread = None
        self.current_mass = 0.5
        self.audio_buffer = deque(maxlen=32)
        
        if PYAUDIO_AVAILABLE:
            self.audio = pyaudio.PyAudio()
        else:
            self.audio = None
    
    def start(self):
        """Start background audio analysis."""
        if not PYAUDIO_AVAILABLE:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._analyze_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop background audio analysis."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _analyze_loop(self):
        """Continuous audio analysis loop."""
        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            while self.running:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float64)
                    
                    # Calculate RMS energy
                    rms = np.sqrt(np.mean(audio_data ** 2)) / 32768.0
                    
                    # Smooth with buffer
                    self.audio_buffer.append(rms)
                    self.current_mass = min(1.0, np.mean(self.audio_buffer) * 3)
                    
                except Exception:
                    pass
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"Audio error: {e}")
    
    def get_mass(self) -> float:
        """Get current emotional mass."""
        return self.current_mass
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop()
        if self.audio:
            self.audio.terminate()


# =============================================================================
# FACE ANALYZER (Geometric Phase)
# =============================================================================

class FaceAnalyzer:
    """Real-time facial geometric phase analysis."""
    
    def __init__(self):
        self.face_cascade = None
        if CV2_AVAILABLE:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def analyze_frame(self, frame: np.ndarray) -> Tuple[float, list, dict]:
        """
        Analyze a video frame for facial geometric phase.
        
        Returns:
            Tuple of (phase, face_rects, tensions)
        """
        if self.face_cascade is None:
            return 0.4, [], {}
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return 0.4, [], {"status": "No face detected"}
        
        # Analyze first face
        x, y, w, h = faces[0]
        face_region = gray[y:y+h, x:x+w]
        
        # Calculate tension from face geometry
        upper_half = face_region[:h//2, :]
        lower_half = face_region[h//2:, :]
        
        brow_tension = np.std(upper_half) / 128.0
        mouth_tension = np.std(lower_half) / 128.0
        
        # Geometric phase
        import math
        y_tension = brow_tension - 0.3
        x_tension = mouth_tension - 0.3
        phi = abs(math.atan2(y_tension, x_tension))
        phi_normalized = min(phi, math.pi / 2)
        
        tensions = {
            "brow": brow_tension,
            "mouth": mouth_tension,
            "phase": phi_normalized
        }
        
        return phi_normalized, list(faces), tensions


# =============================================================================
# VISUAL DISPLAY
# =============================================================================

class EmotionalVisualDisplay:
    """
    Real-time visual display of emotional state analysis.
    
    Shows:
    - Live camera feed
    - Face detection box
    - Emotion overlay
    - Mass/Phase meters
    """
    
    def __init__(self):
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required. Install: pip install opencv-python")
        
        self.camera = cv2.VideoCapture(0)
        self.audio_analyzer = AudioAnalyzer()
        self.face_analyzer = FaceAnalyzer()
        self.api = EmotionalStateAPI()
        
        self.current_emotion = EmotionalState.NEUTRAL
        self.current_mass = 0.5
        self.current_phase = 0.4
        
        # Smoothing
        self.emotion_history = deque(maxlen=10)
    
    def draw_overlay(self, frame: np.ndarray, faces: list, 
                     mass: float, phase: float, emotion: EmotionalState) -> np.ndarray:
        """Draw analysis overlay on frame."""
        
        h, w = frame.shape[:2]
        color = COLORS.get(emotion.value, (200, 200, 200))
        
        # Draw face rectangles
        for (x, y, fw, fh) in faces:
            # Face box
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 3)
            
            # Emotion label above face
            label = f"{emotion.value}"
            label_size = cv2.getTextSize(label, FONT, 1.0, 2)[0]
            cv2.rectangle(frame, (x, y - 40), (x + label_size[0] + 10, y), color, -1)
            cv2.putText(frame, label, (x + 5, y - 10), FONT, 1.0, (0, 0, 0), 2)
        
        # Draw info panel (top-left)
        panel_h = 140
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "CST Emotional Analysis", (20, 35), 
                    FONT, 0.7, (255, 255, 255), 2)
        
        # Mass meter
        cv2.putText(frame, f"Mass: {mass:.2f}", (20, 60), 
                    FONT, 0.5, (200, 200, 200), 1)
        cv2.rectangle(frame, (100, 50), (280, 65), (50, 50, 50), -1)
        cv2.rectangle(frame, (100, 50), (100 + int(180 * mass), 65), (0, 255, 0), -1)
        
        # Phase meter  
        phase_norm = phase / (3.14159 / 2)  # Normalize to 0-1
        cv2.putText(frame, f"Phase: {phase:.2f}", (20, 85), 
                    FONT, 0.5, (200, 200, 200), 1)
        cv2.rectangle(frame, (100, 75), (280, 90), (50, 50, 50), -1)
        cv2.rectangle(frame, (100, 75), (100 + int(180 * phase_norm), 90), (255, 100, 0), -1)
        
        # Emotion result
        emoji = {"HAPPY": ":)", "ANGRY": ":(", "SAD": ":'(", "NEUTRAL": ":|"}.get(emotion.value, "?")
        cv2.putText(frame, f"Emotion: {emoji} {emotion.value}", (20, 120), 
                    FONT, 0.7, color, 2)
        
        # Instructions (bottom)
        cv2.putText(frame, "Press 'q' to quit | 'r' to reset", (10, h - 20), 
                    FONT, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def run(self):
        """Run the visual display loop."""
        
        print("\n" + "=" * 60)
        print("   🎭 CST EMOTIONAL STATE ANALYZER - VISUAL MODE")
        print("   Real-time Face + Audio Analysis")
        print("=" * 60)
        print("\n  Press 'q' to quit\n")
        
        # Start audio analysis
        self.audio_analyzer.start()
        
        try:
            while True:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                # Mirror for natural feel
                frame = cv2.flip(frame, 1)
                
                # Get current measurements
                self.current_mass = self.audio_analyzer.get_mass()
                self.current_phase, faces, tensions = self.face_analyzer.analyze_frame(frame)
                
                # Determine emotion
                self.current_emotion = determine_state(self.current_mass, self.current_phase)
                
                # Smooth emotion (vote over recent history)
                self.emotion_history.append(self.current_emotion)
                emotion_counts = {}
                for e in self.emotion_history:
                    emotion_counts[e] = emotion_counts.get(e, 0) + 1
                display_emotion = max(emotion_counts.keys(), key=lambda e: emotion_counts[e])
                
                # Draw overlay
                frame = self.draw_overlay(
                    frame, faces, 
                    self.current_mass, self.current_phase, 
                    display_emotion
                )
                
                # Show window
                cv2.imshow(WINDOW_NAME, frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.emotion_history.clear()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        self.audio_analyzer.cleanup()
        self.camera.release()
        cv2.destroyAllWindows()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_visual_display():
    """Launch the visual emotional display."""
    if not CV2_AVAILABLE:
        print("❌ OpenCV required for visual display!")
        print("   Install: pip install opencv-python")
        return
    
    display = EmotionalVisualDisplay()
    display.run()


if __name__ == "__main__":
    run_visual_display()
