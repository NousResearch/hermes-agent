"""
real_time_audio_pipe.py – Capture microphone audio, convert to frequency tokens in real time.

The pipeline runs in a background thread, reads raw audio from the default input device,
computes a short‑time Fourier transform (STFT) on each chunk, extracts the most energetic
frequency bins and maps them to symbolic tokens that can be fed to the 12D/42D models.

Design goals:
* Low latency – process ~50 ms frames.
* Minimal CPU impact – use numpy FFT, optional librosa for convenience.
* Thread‑safe – expose a ``deque`` that the training loop can poll each iteration.
* Configurable – sample rate, frame size, number of bins, token vocabulary.
"""

import threading
import time
import math
import numpy as np
import sounddevice as sd
from collections import deque
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------------
# Configuration – tweak as needed
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050          # Hz – matches other audio modules
FRAME_DURATION = 0.05        # seconds per chunk (≈ 50 ms)
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)
TOP_BINS = 16                # Number of frequency bins turned into tokens
TOKEN_VOCAB = [f"<freq_{i}>" for i in range(TOP_BINS)]

def magnitudes_to_tokens(mags: np.ndarray) -> list[str]:
    if mags.size == 0:
        return []
    top_idx = np.argpartition(mags, -TOP_BINS)[-TOP_BINS:]
    top_idx = top_idx[np.argsort(-mags[top_idx])]
    tokens = [TOKEN_VOCAB[i % len(TOKEN_VOCAB)] for i in top_idx]
    return tokens

class RealTimeAudioPipe:
    """Capture microphone audio and expose a rolling token buffer."""
    def __init__(self):
        self._running = threading.Event()
        self._buffer = deque(maxlen=1024)
        self._token_buffer = deque(maxlen=4096)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._stream = None

    def start(self):
        self._running.set()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SIZE,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._stream:
            self._stream.stop()
            self._stream.close()
        self._thread.join(timeout=1.0)

    def pop_tokens(self) -> list[str]:
        tokens = list(self._token_buffer)
        self._token_buffer.clear()
        return tokens

    def _audio_callback(self, indata, frames, time_info, status):
        if not self._running.is_set():
            return
        audio_chunk = indata[:, 0].copy()
        self._buffer.append(audio_chunk)

    def _worker(self):
        phi = 1.618033988749895
        
        while self._running.is_set():
            if not self._buffer:
                time.sleep(0.01)
                continue
            chunk = self._buffer.popleft()
            
            # --- 1. TIME DOMAIN METRICS ---
            rms_raw = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0
            # Scale RMS by 10 to match the 12D HTML frontend visualization
            rms_energy = rms_raw * 10.0
            
            # --- 2. FREQUENCY DOMAIN (FFT) ---
            windowed = chunk * np.hanning(len(chunk))
            # Normalize spectrum to true amplitude [0, 1] range rather than scaling with N
            spectrum = np.abs(np.fft.rfft(windowed)) / (len(chunk) / 2.0)
            freqs = np.fft.rfftfreq(len(chunk), d=1/SAMPLE_RATE)
            
            # --- 3. SPECTRAL CENTROID ---
            weighted_sum = np.sum(freqs * spectrum)
            magnitude_sum = np.sum(spectrum)
            centroid = float(weighted_sum / magnitude_sum) if magnitude_sum > 0 else 0.0
            
            # --- 4. TOP 10 FREQUENCIES ---
            num_bins = min(10, len(spectrum))
            if num_bins > 0:
                top_idx = np.argpartition(spectrum, -num_bins)[-num_bins:]
                top_idx = top_idx[np.argsort(-spectrum[top_idx])]
                
                top_freqs = []
                for idx in top_idx:
                    mag = float(spectrum[idx])
                    if mag > 0.001: # Lower noise floor filter, scaled for amplitude
                        top_freqs.append({
                            "frequency": float(freqs[idx]),
                            "magnitude": mag
                        })
                
                # --- 5. PHI-HARMONICS (Golden Ratio Resonance) ---
                harmonics = []
                if len(top_freqs) > 0:
                    fundamental = top_freqs[0]["frequency"]
                    if fundamental > 0:
                        for n in range(1, 9):
                            h_freq = fundamental * math.pow(phi, n / 2.0)
                            harmonics.append(float(h_freq))
                
                # --- 6. TOKEN GENERATION ---
                # Bundle the full acoustic state into a structured payload
                token_payload = {
                    "timestamp": time.time(),
                    "rms_energy": rms_energy,
                    "spectral_centroid": centroid,
                    "top_frequencies": top_freqs,
                    "phi_harmonics": harmonics
                }
                
                # Push the structured token state instead of raw strings
                self._token_buffer.append(token_payload)
            
            # Sleep a tiny bit to keep CPU usage low
            time.sleep(0.001)

# ---------------------------------------------------------------------------
# Simple sanity‑check when run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pipe = RealTimeAudioPipe()
    print("Starting real‑time audio capture – press Ctrl+C to stop")
    pipe.start()
    try:
        while True:
            toks = pipe.pop_tokens()
            if toks:
                print("Tokens:", toks)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Stopping…")
        pipe.stop()

