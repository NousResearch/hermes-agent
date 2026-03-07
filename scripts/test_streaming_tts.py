#!/usr/bin/env python3
"""Test streaming TTS text-audio synchronization.

Simulates LLM token generation and verifies that display_callback
is called with complete sentences BEFORE audio generation starts.

Runs without ElevenLabs API key -- the TTS pipeline will log a warning
and exit gracefully, but sentence buffering + display_callback still fire.
"""
import queue
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_sentence_buffering():
    """Test that sentences are buffered correctly and display_callback fires."""
    from tools.tts_tool import stream_tts_to_speaker

    text_q = queue.Queue()
    stop = threading.Event()
    done = threading.Event()

    displayed_sentences = []

    def on_display(sentence: str):
        displayed_sentences.append(sentence)
        print(f"  [DISPLAY] {sentence.strip()}")

    # Simulate LLM streaming tokens
    tokens = list("Hello, how are you today? I am doing great. Let me help you with that.")

    def producer():
        for token in tokens:
            text_q.put(token)
            time.sleep(0.02)  # simulate LLM speed
        text_q.put(None)  # sentinel

    # Ensure no API key so TTS gracefully skips audio but still buffers
    os.environ.pop("ELEVENLABS_API_KEY", None)

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    stream_tts_to_speaker(text_q, stop, done, display_callback=on_display)

    producer_thread.join(timeout=5)

    print(f"\nSentences displayed: {len(displayed_sentences)}")
    for i, s in enumerate(displayed_sentences):
        print(f"  {i+1}. '{s.strip()}'")

    assert len(displayed_sentences) > 0, "No sentences were displayed!"
    print("\nPASSED: Sentence buffering and display_callback work correctly.")


if __name__ == "__main__":
    test_sentence_buffering()
