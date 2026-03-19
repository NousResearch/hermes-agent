"""
VIDEO LISTENER (AUDITORY INPUT)
===============================
Extracts vibrational information from video/audio files
and feeds it into the 12D Auditory Cortex.
"""

import sys
import os
from pathlib import Path
import librosa
import numpy as np
import pickle
import argparse

# Add path to engine
sys.path.append(str(Path(__file__).parent))
from audio_engine import AudioLearningSystem

def listen_to_file(file_path, memory_file="auditory_memory.pkl"):
    print(f"👂 Listening to: {file_path}")
    
    # Initialize System
    system = AudioLearningSystem(sample_rate=22050) # Standard for ML audio
    
    # Load Memory if exists
    if os.path.exists(memory_file):
        print(f"🧠 Loading auditory memory from {memory_file}...")
        try:
            with open(memory_file, "rb") as f:
                saved_data = pickle.load(f)
                system.memory.tokens = saved_data['tokens']
                # Rebuild FAISS index
                for token in system.memory.tokens:
                    vec = token.embedding.to_vector().astype('float32').reshape(1, -1)
                    system.memory.index.add(vec)
                    system.memory.index_to_token[system.memory.index.ntotal - 1] = token
            print(f"   Restored {len(system.memory.tokens)} memories.")
        except Exception as e:
            print(f"   Warning: Could not load memory ({e}). Starting fresh.")

    # Load Audio
    try:
        # librosa.load can often handle video files if ffmpeg is installed
        audio, sr = librosa.load(file_path, sr=system.sample_rate)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    print(f"   Audio loaded: {len(audio)/sr:.2f} seconds")
    
    # Process
    print("   Processing vibrational patterns...")
    chunk_size = int(system.sample_rate * 1.0) # 1 second chunks
    total_tokens = 0
    
    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i:i+chunk_size]
        new_tokens = system.process_audio(chunk)
        total_tokens += len(new_tokens)
        
        if i % (chunk_size * 10) == 0:
            print(f"   ...absorbed {total_tokens} patterns", end="\r")
            
    print(f"\n✅ Finished. Absorbed {total_tokens} new patterns.")
    print(f"   Total Memory Size: {len(system.memory.tokens)}")
    
    # Save Memory
    print(f"💾 Saving memory to {memory_file}...")
    with open(memory_file, "wb") as f:
        pickle.dump({
            'tokens': system.memory.tokens,
            'lorenz': (system.lorenz.x, system.lorenz.y, system.lorenz.z)
        }, f)
    print("   Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="12D Video Listener")
    parser.add_argument("file", help="Path to video or audio file")
    args = parser.parse_args()
    
    listen_to_file(args.file)
