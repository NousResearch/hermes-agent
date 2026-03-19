"""
Emotional State API - Live Capture Module

Real-time microphone and camera capture for 12D CST Emotional Analysis.
This module provides live input capabilities for the EmotionalStateAPI.
"""

import io
import tempfile
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

# Audio capture
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

# Video capture
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# For saving audio
try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class LiveCapture:
    """
    Real-time capture from microphone and camera for CST emotional analysis.
    
    Usage:
        capture = LiveCapture()
        audio_data, image_data = capture.capture_both(duration=3.0)
    """
    
    # Audio settings
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 1024
    FORMAT = None  # Set in __init__ if pyaudio available
    
    def __init__(self):
        """Initialize live capture devices."""
        self.audio = None
        self.camera = None
        
        if PYAUDIO_AVAILABLE:
            self.FORMAT = pyaudio.paInt16
            self.audio = pyaudio.PyAudio()
        
        if CV2_AVAILABLE:
            self.camera = cv2.VideoCapture(0)
    
    def capture_audio(self, duration: float = 3.0) -> Tuple[Optional[str], dict]:
        """
        Capture audio from microphone.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Tuple of (temp_wav_path, metadata)
        """
        if not PYAUDIO_AVAILABLE or self.audio is None:
            return None, {"error": "PyAudio not available. Install with: pip install pyaudio"}
        
        print(f"🎤 Recording for {duration} seconds...")
        
        try:
            stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            frames = []
            num_chunks = int(self.SAMPLE_RATE / self.CHUNK_SIZE * duration)
            
            for i in range(num_chunks):
                data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)
                
                # Progress indicator
                progress = int((i + 1) / num_chunks * 20)
                print(f"\r  [{'=' * progress}{' ' * (20 - progress)}] {int((i + 1) / num_chunks * 100)}%", end="")
            
            print("\n  ✓ Recording complete!")
            
            stream.stop_stream()
            stream.close()
            
            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            if SCIPY_AVAILABLE:
                wavfile.write(temp_path, self.SAMPLE_RATE, audio_data)
            else:
                # Manual WAV writing
                import wave
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.SAMPLE_RATE)
                    wf.writeframes(audio_data.tobytes())
            
            return temp_path, {
                "duration": duration,
                "sample_rate": self.SAMPLE_RATE,
                "samples": len(audio_data),
                "peak_amplitude": float(np.max(np.abs(audio_data))) / 32768
            }
            
        except Exception as e:
            return None, {"error": f"Recording failed: {str(e)}"}
    
    def capture_image(self) -> Tuple[Optional[str], dict]:
        """
        Capture image from camera.
        
        Returns:
            Tuple of (temp_image_path, metadata)
        """
        if not CV2_AVAILABLE:
            return None, {"error": "OpenCV not available. Install with: pip install opencv-python"}
        
        if self.camera is None or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
            time.sleep(0.5)  # Warm-up
        
        if not self.camera.isOpened():
            return None, {"error": "Camera not available"}
        
        print("📷 Capturing image...")
        
        # Take a few frames to stabilize
        for _ in range(5):
            ret, frame = self.camera.read()
        
        if not ret or frame is None:
            return None, {"error": "Failed to capture frame"}
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        cv2.imwrite(temp_path, frame)
        
        print("  ✓ Image captured!")
        
        return temp_path, {
            "width": frame.shape[1],
            "height": frame.shape[0],
            "channels": frame.shape[2] if len(frame.shape) > 2 else 1
        }
    
    def capture_both(self, audio_duration: float = 3.0) -> Tuple[Optional[str], Optional[str], dict]:
        """
        Capture both audio and image simultaneously.
        
        Args:
            audio_duration: Audio recording duration
            
        Returns:
            Tuple of (audio_path, image_path, combined_metadata)
        """
        print("\n" + "=" * 50)
        print("  LIVE CAPTURE - 12D CST Emotional Analysis")
        print("=" * 50)
        
        # Capture image first (quick)
        image_path, image_meta = self.capture_image()
        
        # Then capture audio
        audio_path, audio_meta = self.capture_audio(audio_duration)
        
        return audio_path, image_path, {
            "audio": audio_meta,
            "image": image_meta
        }
    
    def cleanup(self):
        """Release resources."""
        if self.audio:
            self.audio.terminate()
        if self.camera:
            self.camera.release()
    
    def __del__(self):
        """Destructor to cleanup resources."""
        self.cleanup()


def check_devices() -> dict:
    """Check available capture devices."""
    status = {
        "pyaudio_available": PYAUDIO_AVAILABLE,
        "opencv_available": CV2_AVAILABLE,
        "microphone": False,
        "camera": False
    }
    
    # Check microphone
    if PYAUDIO_AVAILABLE:
        try:
            audio = pyaudio.PyAudio()
            if audio.get_device_count() > 0:
                status["microphone"] = True
            audio.terminate()
        except:
            pass
    
    # Check camera
    if CV2_AVAILABLE:
        try:
            cam = cv2.VideoCapture(0)
            if cam.isOpened():
                status["camera"] = True
            cam.release()
        except:
            pass
    
    return status


if __name__ == "__main__":
    print("Live Capture Device Check")
    print("-" * 30)
    
    status = check_devices()
    
    print(f"  PyAudio:    {'✓' if status['pyaudio_available'] else '✗'}")
    print(f"  OpenCV:     {'✓' if status['opencv_available'] else '✗'}")
    print(f"  Microphone: {'✓ Ready' if status['microphone'] else '✗ Not found'}")
    print(f"  Camera:     {'✓ Ready' if status['camera'] else '✗ Not found'}")
    
    if not PYAUDIO_AVAILABLE:
        print("\n  Install PyAudio: pip install pyaudio")
    if not CV2_AVAILABLE:
        print("\n  Install OpenCV: pip install opencv-python")
