"""
cosmos YouTube Intelligence - Video Insight Extraction.

"I can tell you exactly what that 10-hour lecture said in 30 seconds."

This module handles YouTube transcripts and video metadata extraction.
Uses youtube-transcript-api for transcript fetching (no API key required).
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from loguru import logger

# Try to import youtube_transcript_api
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
    TRANSCRIPT_API_AVAILABLE = True
except ImportError:
    TRANSCRIPT_API_AVAILABLE = False
    logger.warning("youtube-transcript-api not installed. Run: pip install youtube-transcript-api")


@dataclass
class TranscriptSegment:
    """A segment of video transcript."""
    text: str
    start: float
    duration: float


@dataclass
class VideoTranscript:
    """Full video transcript with metadata."""
    video_id: str
    title: Optional[str]
    segments: List[TranscriptSegment]
    language: str
    is_auto_generated: bool

    @property
    def full_text(self) -> str:
        """Get complete transcript as single string."""
        return " ".join(seg.text for seg in self.segments)

    @property
    def duration_seconds(self) -> float:
        """Estimated video duration from transcript."""
        if not self.segments:
            return 0
        last = self.segments[-1]
        return last.start + last.duration

    def get_text_at_time(self, seconds: float, window: float = 30) -> str:
        """Get transcript text around a specific timestamp."""
        relevant = [
            seg for seg in self.segments
            if abs(seg.start - seconds) <= window
        ]
        return " ".join(seg.text for seg in relevant)


@dataclass
class VideoMetadata:
    """Video metadata from various sources."""
    video_id: str
    title: str
    channel: Optional[str] = None
    description: Optional[str] = None
    duration: Optional[str] = None
    view_count: Optional[int] = None
    publish_date: Optional[datetime] = None
    thumbnail_url: Optional[str] = None


class YouTubeSkill:
    """
    YouTube video intelligence skill.

    Capabilities:
    - Fetch transcripts (auto-generated or manual)
    - Extract video metadata
    - Search for videos (via web scraping)
    - Summarize video content
    """

    def __init__(self, llm_fn=None):
        """
        Initialize YouTube skill.

        Args:
            llm_fn: Optional LLM function for summarization
        """
        self.llm_fn = llm_fn
        self._transcript_cache: Dict[str, VideoTranscript] = {}

    def extract_id(self, url: str) -> Optional[str]:
        """
        Extract Video ID from various YouTube URL formats.

        Supports:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - https://www.youtube.com/v/VIDEO_ID
        """
        patterns = [
            r"(?:v=|\/v\/|youtu\.be\/|\/embed\/)([0-9A-Za-z_-]{11})",
            r"^([0-9A-Za-z_-]{11})$",  # Just the ID itself
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    async def get_transcript(
        self,
        video_id_or_url: str,
        languages: List[str] = None,
        include_auto_generated: bool = True,
    ) -> VideoTranscript:
        """
        Fetch transcript for a YouTube video.

        Args:
            video_id_or_url: YouTube video ID or URL
            languages: Preferred languages (default: ['en'])
            include_auto_generated: Include auto-generated captions

        Returns:
            VideoTranscript object

        Raises:
            ValueError: If video ID cannot be extracted
            RuntimeError: If transcript cannot be fetched
        """
        # Extract video ID
        video_id = self.extract_id(video_id_or_url) or video_id_or_url

        if not video_id or len(video_id) != 11:
            raise ValueError(f"Invalid video ID or URL: {video_id_or_url}")

        # Check cache
        if video_id in self._transcript_cache:
            logger.debug(f"YouTube: Using cached transcript for {video_id}")
            return self._transcript_cache[video_id]

        if not TRANSCRIPT_API_AVAILABLE:
            raise RuntimeError(
                "youtube-transcript-api not installed. "
                "Run: pip install youtube-transcript-api"
            )

        languages = languages or ['en', 'en-US', 'en-GB']
        logger.info(f"YouTube: Fetching transcript for {video_id}")

        try:
            # Try to get transcript
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Try manual transcripts first
            transcript = None
            is_auto = False

            try:
                transcript = transcript_list.find_manually_created_transcript(languages)
            except NoTranscriptFound:
                if include_auto_generated:
                    try:
                        transcript = transcript_list.find_generated_transcript(languages)
                        is_auto = True
                    except NoTranscriptFound:
                        pass

            if transcript is None:
                # Try any available transcript and translate
                try:
                    available = list(transcript_list)
                    if available:
                        transcript = available[0]
                        if transcript.language_code not in languages:
                            transcript = transcript.translate('en')
                        is_auto = transcript.is_generated
                except Exception as e:
                    logger.debug(f"Fallback transcript translation failed: {e}")

            if transcript is None:
                raise RuntimeError(f"No transcript available for video {video_id}")

            # Fetch the actual transcript data
            transcript_data = transcript.fetch()

            segments = [
                TranscriptSegment(
                    text=item['text'],
                    start=item['start'],
                    duration=item['duration'],
                )
                for item in transcript_data
            ]

            result = VideoTranscript(
                video_id=video_id,
                title=None,  # Would need separate API call
                segments=segments,
                language=transcript.language_code,
                is_auto_generated=is_auto,
            )

            # Cache the result
            self._transcript_cache[video_id] = result

            logger.info(
                f"YouTube: Got transcript for {video_id} "
                f"({len(segments)} segments, {result.duration_seconds:.0f}s, "
                f"{'auto' if is_auto else 'manual'})"
            )

            return result

        except TranscriptsDisabled:
            raise RuntimeError(f"Transcripts are disabled for video {video_id}")
        except VideoUnavailable:
            raise RuntimeError(f"Video {video_id} is unavailable")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch transcript: {e}")

    async def get_transcript_text(self, video_id_or_url: str) -> str:
        """
        Get transcript as plain text.

        Convenience method that returns just the text content.
        """
        transcript = await self.get_transcript(video_id_or_url)
        return transcript.full_text

    async def summarize_video(
        self,
        video_id_or_url: str,
        max_length: int = 500,
    ) -> Dict[str, Any]:
        """
        Summarize a video's content.

        Args:
            video_id_or_url: YouTube video ID or URL
            max_length: Maximum summary length

        Returns:
            Dict with summary and metadata
        """
        transcript = await self.get_transcript(video_id_or_url)
        full_text = transcript.full_text

        summary = full_text[:max_length] + "..." if len(full_text) > max_length else full_text

        # Use LLM if available
        if self.llm_fn and len(full_text) > 100:
            try:
                import asyncio
                prompt = f"""Summarize this video transcript in {max_length} characters or less.
Focus on the main points and key takeaways.

Transcript:
{full_text[:8000]}

Summary:"""
                if asyncio.iscoroutinefunction(self.llm_fn):
                    summary = await self.llm_fn(prompt)
                else:
                    summary = self.llm_fn(prompt)
                summary = summary.strip()[:max_length]
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}")

        return {
            "video_id": transcript.video_id,
            "summary": summary,
            "duration_seconds": transcript.duration_seconds,
            "language": transcript.language,
            "is_auto_generated": transcript.is_auto_generated,
            "word_count": len(full_text.split()),
        }

    async def search_videos(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for YouTube videos.

        Uses web scraping since YouTube Data API requires authentication.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of video metadata dicts
        """
        import urllib.parse
        import urllib.request

        logger.info(f"YouTube: Searching for '{query}'")

        try:
            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://www.youtube.com/results?search_query={encoded_query}"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            req = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(req, timeout=10) as response:
                html = response.read().decode("utf-8")

            # Extract video IDs from the page
            video_ids = re.findall(r'"videoId":"([A-Za-z0-9_-]{11})"', html)
            # Deduplicate while preserving order
            seen = set()
            unique_ids = []
            for vid in video_ids:
                if vid not in seen:
                    seen.add(vid)
                    unique_ids.append(vid)

            results = []
            for vid in unique_ids[:limit]:
                results.append({
                    "video_id": vid,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "title": f"Video {vid}",  # Would need additional parsing
                })

            if results:
                logger.info(f"YouTube: Found {len(results)} videos")
                return results

        except Exception as e:
            logger.warning(f"YouTube search failed: {e}")

        # Fallback
        return [{
            "video_id": "search_failed",
            "url": f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}",
            "title": f"Search YouTube for: {query}",
            "note": "Direct search results unavailable",
        }]

    async def get_chapters(self, video_id_or_url: str) -> List[Dict[str, Any]]:
        """
        Extract chapter markers from video transcript.

        Analyzes transcript for natural topic breaks.
        """
        transcript = await self.get_transcript(video_id_or_url)

        if not transcript.segments:
            return []

        # Simple chapter detection: significant pauses or topic changes
        chapters = []
        chapter_interval = max(60, transcript.duration_seconds / 10)  # At least 1 min chapters

        current_start = 0
        for i, seg in enumerate(transcript.segments):
            if seg.start - current_start >= chapter_interval:
                chapters.append({
                    "start": current_start,
                    "end": seg.start,
                    "text": transcript.get_text_at_time(current_start, window=chapter_interval / 2)[:200],
                })
                current_start = seg.start

        # Add final chapter
        if transcript.segments:
            chapters.append({
                "start": current_start,
                "end": transcript.duration_seconds,
                "text": transcript.get_text_at_time(current_start, window=chapter_interval / 2)[:200],
            })

        return chapters

    def clear_cache(self):
        """Clear transcript cache."""
        self._transcript_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_videos": len(self._transcript_cache),
            "video_ids": list(self._transcript_cache.keys()),
        }


# Global instance for convenience
youtube_skill = YouTubeSkill()
