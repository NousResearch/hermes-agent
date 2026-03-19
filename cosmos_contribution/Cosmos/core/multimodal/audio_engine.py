"""
12D AUDIO ENGINE (AUDITORY CORTEX)
==================================
Physics-based audio embedding and learning system.
Author: Cory Shane Davis (Implemented by Antigravity)
"""

import numpy as np
import librosa
import faiss
from dataclasses import dataclass
from typing import Tuple, Optional
import hashlib
import uuid
from datetime import datetime
import os

# ============================================================================
# CONSTANTS
# ============================================================================

PHI = 1.618033988749895
C = 299792458
G_SCALED = 1e-10
EPSILON = 0.1

# ============================================================================
# AUDIO PROCESSING
# ============================================================================

def compute_fft(audio, sample_rate):
    """FFT with magnitude and phase"""
    N = len(audio)
    window = np.hanning(N)
    windowed = audio * window
    fft_result = np.fft.rfft(windowed)
    
    magnitudes = 20 * np.log10(np.abs(fft_result) + 1e-10)
    phases = np.angle(fft_result)
    frequencies = np.fft.rfftfreq(N, 1.0/sample_rate)
    
    return frequencies, magnitudes, phases

def extract_fundamental_frequency(audio, sample_rate):
    """Extract f0 using YIN"""
    try:
        f0 = librosa.yin(audio, fmin=80, fmax=800, sr=sample_rate)
        return np.median(f0[f0 > 0]) if any(f0 > 0) else None
    except:
        return None

def spectral_centroid(audio, sample_rate):
    """Compute spectral centroid"""
    freqs, mags, _ = compute_fft(audio, sample_rate)
    mags_linear = 10 ** (mags / 20)
    if np.sum(mags_linear) == 0: return 0
    return np.sum(freqs * mags_linear) / np.sum(mags_linear)

def spectral_spread(audio, sample_rate):
    """Compute spectral spread"""
    freqs, mags, _ = compute_fft(audio, sample_rate)
    mags_linear = 10 ** (mags / 20)
    centroid = spectral_centroid(audio, sample_rate)
    if np.sum(mags_linear) == 0: return 0
    return np.sqrt(np.sum((freqs - centroid)**2 * mags_linear) / np.sum(mags_linear))

def extract_mfccs(audio, sample_rate, n_mfcc=13):
    """Extract MFCCs"""
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        return np.mean(mfccs, axis=1)
    except:
        return np.zeros(n_mfcc)

def spectral_entropy(audio, sample_rate):
    """Compute spectral entropy"""
    _, mags, _ = compute_fft(audio, sample_rate)
    power = 10 ** (mags / 10)
    if np.sum(power) == 0: return 0
    prob = power / np.sum(power)
    entropy = -np.sum(prob * np.log2(prob + 1e-10))
    max_entropy = np.log2(len(prob)) if len(prob) > 0 else 1
    return entropy / max_entropy

def generate_phi_harmonics(fundamental, num_harmonics=12):
    """Generate φ-harmonic series"""
    harmonics = []
    for n in range(num_harmonics):
        exponent = n - (num_harmonics / 2)
        freq = fundamental * (PHI ** exponent)
        
        # Octave folding
        while freq > fundamental * 4:
            freq /= 2
        while freq < fundamental / 2:
            freq *= 2
        
        # 12-TET quantization
        midi = 69 + 12 * np.log2(freq / 440.0)
        quantized_midi = round(midi)
        final_freq = 440 * (2 ** ((quantized_midi - 69) / 12))
        
        harmonics.append(final_freq)
    
    return sorted(harmonics)

def segment_audio(audio, sample_rate):
    """Segment audio at onsets"""
    try:
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sample_rate, units='samples')
        
        segments = []
        for i in range(len(onset_frames)):
            start = onset_frames[i]
            end = onset_frames[i+1] if i+1 < len(onset_frames) else len(audio)
            segments.append(audio[start:end])
        
        # If no onsets found, return whole chunk if small enough, or split
        if not segments:
            segments = [audio]
            
        return segments
    except:
        return [audio]

# ============================================================================
# 12D EMBEDDING
# ============================================================================

@dataclass
class Audio12DEmbedding:
    """12-dimensional audio embedding"""
    D1_energy: float
    D2_mass: float
    D3_phi_coupling: float
    D4_chaos: float
    D5_vx: float
    D6_vy: float
    D7_vz: float
    D8_connectivity: float
    D9_cosmic_energy: float
    D10_entropy: float
    D11_frequency: float
    D12_x12: float
    
    def to_vector(self):
        return np.array([
            self.D1_energy, self.D2_mass, self.D3_phi_coupling, self.D4_chaos,
            self.D5_vx, self.D6_vy, self.D7_vz, self.D8_connectivity,
            self.D9_cosmic_energy, self.D10_entropy, self.D11_frequency, self.D12_x12
        ])

def create_12d_embedding(audio, sample_rate):
    """Create 12D embedding from audio"""
    # D1: Energy
    rms = np.sqrt(np.mean(audio ** 2))
    D1 = rms ** 2
    
    # D2: Mass-energy
    D2 = (PHI * D1) / (C ** 2) * 1e17
    
    # D3: Phi coupling
    D3 = PHI * D1
    
    # D4: Chaos (spectral entropy)
    D4 = spectral_entropy(audio, sample_rate)
    
    # D5-D7: Velocity (placeholder - needs temporal context)
    D5, D6, D7 = 0.0, 0.0, 0.0
    
    # D8: Connectivity (placeholder - updated later)
    D8 = 0.5
    
    # D9: Cosmic energy (spectral centroid)
    centroid = spectral_centroid(audio, sample_rate)
    D9 = centroid / (sample_rate / 2)
    
    # D10: Entropy (spectral spread)
    spread = spectral_spread(audio, sample_rate)
    D10 = spread / (sample_rate / 2)
    
    # D11: Frequency
    f0 = extract_fundamental_frequency(audio, sample_rate)
    D11 = np.log10(f0 / 20) / np.log10(20000 / 20) if f0 and f0 > 0 else 0
    
    # D12: Adaptive state (initial)
    D12 = 0.0
    
    return Audio12DEmbedding(D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12)

# ============================================================================
# LIGHT TOKEN
# ============================================================================

class MusicalLightToken:
    """Complete light token"""
    
    def __init__(self, audio, sample_rate):
        self.token_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
        
        # Audio data
        self.audio = audio
        self.sample_rate = sample_rate
        self.duration = len(audio) / sample_rate
        
        # Features
        self.fundamental_hz = extract_fundamental_frequency(audio, sample_rate)
        self.loudness_db = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-10)
        self.mfcc = extract_mfccs(audio, sample_rate)
        
        # Spectral signature
        freqs, mags, phases = compute_fft(audio, sample_rate)
        self.spectral_signature = {'frequencies': freqs, 'magnitudes': mags, 'phases': phases}
        
        # 12D embedding
        self.embedding = create_12d_embedding(audio, sample_rate)
        
        # φ-harmonics
        if self.fundamental_hz:
            self.phi_harmonics = generate_phi_harmonics(self.fundamental_hz)
        else:
            self.phi_harmonics = []
        
        # Hash
        self.hash = hashlib.sha256(
            f"{self.token_id}{self.timestamp}{self.fundamental_hz}".encode()
        ).hexdigest()
    
    def spectral_similarity(self, other):
        """Cosine similarity in spectral domain"""
        mag1 = self.spectral_signature['magnitudes']
        mag2 = other.spectral_signature['magnitudes']
        
        min_len = min(len(mag1), len(mag2))
        mag1 = mag1[:min_len]
        mag2 = mag2[:min_len]
        
        dot = np.dot(mag1, mag2)
        norm = np.linalg.norm(mag1) * np.linalg.norm(mag2)
        
        return dot / (norm + 1e-10)

# ============================================================================
# SPECTRAL MEMORY
# ============================================================================

class SpectralMemory:
    """Vector database for tokens"""
    
    def __init__(self):
        self.tokens = []
        self.index = faiss.IndexFlatL2(12)
        self.index_to_token = {}
    
    def add_token(self, token):
        self.tokens.append(token)
        
        vec = token.embedding.to_vector().astype('float32').reshape(1, -1)
        idx = self.index.ntotal
        self.index.add(vec)
        self.index_to_token[idx] = token
    
    def find_similar(self, query_token, k=5):
        if self.index.ntotal == 0: return []
        
        vec = query_token.embedding.to_vector().astype('float32').reshape(1, -1)
        distances, indices = self.index.search(vec, k)
        
        return [
            (self.index_to_token[idx], dist)
            for idx, dist in zip(indices[0], distances[0])
            if idx in self.index_to_token
        ]

# ============================================================================
# LORENZ ATTRACTOR
# ============================================================================

class LorenzAttractor:
    """Chaos dynamics"""
    
    def __init__(self, sigma=10, rho=28, beta=8/3, dt=0.01):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.x, self.y, self.z = 1.0, 1.0, 1.0
    
    def step(self):
        dx = self.sigma * (self.y - self.x) * self.dt
        dy = (self.x * (self.rho - self.z) - self.y) * self.dt
        dz = (self.x * self.y - self.beta * self.z) * self.dt
        
        self.x += dx
        self.y += dy
        self.z += dz

# ============================================================================
# COMPLETE SYSTEM
# ============================================================================

class AudioLearningSystem:
    """Complete 12D audio learning system"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.memory = SpectralMemory()
        self.lorenz = LorenzAttractor()
        
        # Learning params
        self.k = 1.0
        self.gamma = 0.005
        self.sigma = 0.5
        self.dt = 0.1
    
    def process_audio(self, audio_chunk):
        """Process audio and create tokens"""
        # Ensure audio is float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
            if np.max(np.abs(audio_chunk)) > 1.0:
                audio_chunk /= 32768.0 # Normalize 16-bit PCM
                
        segments = segment_audio(audio_chunk, self.sample_rate)
        
        new_tokens = []
        for segment in segments:
            if len(segment) < 100:
                continue
            
            token = MusicalLightToken(segment, self.sample_rate)
            
            # Update connectivity
            if len(self.memory.tokens) > 0:
                similar = self.memory.find_similar(token, k=10)
                if similar:
                    token.embedding.D8_connectivity = np.mean([1.0 / (d + 1) for _, d in similar])
            
            # Evolve x₁₂
            omega = token.embedding.D8_connectivity
            dx12 = self.k * omega - self.gamma * token.embedding.D12_x12
            token.embedding.D12_x12 = np.clip(token.embedding.D12_x12 + dx12 * self.dt, -1, 1)
            
            self.memory.add_token(token)
            new_tokens.append(token)
        
        self.lorenz.step()
        return new_tokens
    
    def generate_response(self, user_audio):
        """Generate audio based on learned patterns"""
        user_tokens = self.process_audio(user_audio)
        
        if not user_tokens:
            return None
        
        query = user_tokens[-1]
        similar = self.memory.find_similar(query, k=5)
        
        if similar:
            fundamentals = [t.fundamental_hz for t, _ in similar if t.fundamental_hz]
            if fundamentals:
                target_f0 = np.mean(fundamentals)
                harmonics = generate_phi_harmonics(target_f0, num_harmonics=6)
                
                # Synthesize
                duration = 1.0
                t = np.linspace(0, duration, int(self.sample_rate * duration))
                audio = np.zeros_like(t)
                
                for i, freq in enumerate(harmonics):
                    amp = (PHI ** -i) / len(harmonics)
                    audio += amp * np.sin(2 * np.pi * freq * t)
                
                audio = audio / np.max(np.abs(audio))
                return audio
        
        return None
