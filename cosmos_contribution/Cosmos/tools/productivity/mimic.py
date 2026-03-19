"""
cosmos Mimic - Voice Clone & TTS Engine
-------------------------------------------

"Good news everyone! I can now speak in my own voice!"

Simple wrapper for system TTS or Coqui AI.
"""

import sys
from loguru import logger

class Mimic:
    def __init__(self, voice_id: str = "default"):
        self.voice_id = voice_id

    def speak(self, text: str):
        """text-to-speech output."""
        logger.info(f"🗣️ Mimic: {text}")
        
        # Cross-platform fallback
        if sys.platform == "darwin":
            import os
            os.system(f'say "{text}"')
        elif sys.platform == "win32":
            # Powershell Speech
            import subprocess
            cmd = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'
            subprocess.Popen(["powershell", cmd])
        else:
            # Linux (espeak)
            import os
            os.system(f'espeak "{text}"')

mimic = Mimic()
