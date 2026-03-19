"""
cosmos Meeting Whisperer
----------------------------

"Did he say 'To shreds'? Oh my."

Simulates a real-time transcript analyzer.
"""

from loguru import logger

class MeetingWhisperer:
    def analyze_segment(self, text: str):
        """Analyze live text for actionable items."""
        text = text.lower()
        if "action item" in text or "todo" in text:
            logger.info(f"📝 ACTION ITEM DETECTED: {text}")
        if "cosmos" in text:
            logger.info("👀 They are talking about me!")

whisperer = MeetingWhisperer()
