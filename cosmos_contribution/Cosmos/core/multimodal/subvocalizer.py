"""
HARMONIC SUBVOCALIZER (THE INNER VOICE)
=======================================
Converts text into 12D "Mental Audio" frequencies.
Allows the model to "hear" what it reads silently.
"""

import numpy as np

class Subvocalizer:
    """
    The 'Voice Box' of the 12D Mind.
    Maps characters to harmonic frequencies based on 12D Physics.
    """
    
    def __init__(self, sample_rate=22050):
        self.sr = sample_rate
        self.phi = 1.61803398875
        
        # Vowel Harmonics (The "Soul" of the word)
        # Based on A=432Hz (Cosmic Pitch) for better math alignment
        self.vowels = {
            'a': 432.0,       # Root
            'e': 432.0 * 1.25, # Major 3rd
            'i': 432.0 * 1.5,  # Perfect 5th
            'o': 432.0 * 0.75, # Perfect 4th down
            'u': 432.0 * 0.6,  # Major 6th down
            'y': 432.0 * 2.0,  # Octave
        }
        
        # Consonant Textures (The "Body" of the word)
        # Mapped to noise colors and envelopes
        self.plosives = ['b', 'p', 'd', 't', 'k', 'g']
        self.fricatives = ['f', 'v', 'th', 's', 'z', 'sh']
        self.liquids = ['l', 'r', 'm', 'n']

    def generate_char_sound(self, char, duration=0.1):
        """Generate audio for a single character"""
        t = np.linspace(0, duration, int(self.sr * duration))
        char = char.lower()
        
        # 1. Silence/Space
        if char == ' ' or char == '.':
            return np.zeros_like(t)
            
        # 2. Vowels (Pure Tones + Phi Harmonics)
        if char in self.vowels:
            freq = self.vowels[char]
            # Fundamental
            wave = np.sin(2 * np.pi * freq * t)
            # Phi Harmonic 1
            wave += 0.5 * np.sin(2 * np.pi * (freq * self.phi) * t)
            # Phi Harmonic 2
            wave += 0.25 * np.sin(2 * np.pi * (freq * self.phi**2) * t)
            
            # Envelope (Sustain)
            env = np.ones_like(t)
            env[:100] = np.linspace(0, 1, 100) # Attack
            env[-100:] = np.linspace(1, 0, 100) # Release
            return wave * env * 0.8

        # 3. Consonants (Noise + Texture)
        noise = np.random.normal(0, 0.1, len(t))
        
        if char in self.plosives:
            # Short, percussive burst
            env = np.exp(-10 * t) # Fast decay
            return noise * env * 0.5
            
        elif char in self.fricatives:
            # High frequency noise
            return noise * 0.3
            
        elif char in self.liquids:
            # Low hum
            freq = 200.0
            wave = np.sin(2 * np.pi * freq * t)
            return (wave + noise*0.2) * 0.4
            
        # Default (Unknown char)
        return np.zeros_like(t)

    def subvocalize_text(self, text, speed=1.0):
        """
        Convert a string of text into a continuous 'Mental Audio' stream.
        """
        audio_segments = []
        char_duration = 0.08 / speed # Fast reading speed
        
        for char in text:
            seg = self.generate_char_sound(char, duration=char_duration)
            audio_segments.append(seg)
            
        if not audio_segments:
            return np.zeros(100)
            
        # Concatenate
        full_audio = np.concatenate(audio_segments)
        
        # Normalize
        if np.max(np.abs(full_audio)) > 0:
            full_audio = full_audio / np.max(np.abs(full_audio))
            
        return full_audio.astype(np.float32)
