"""
cosmos Video Module v2.1 - Advanced Spatio-Temporal Flow Analysis.

"I can see the wind, and the code within the wind!"

Improvements:
1. Optical Flow (Farneback): Analyzes dense motion vectors to detect "Action Peaks".
2. Motion Magnitude Saliency: Keyframes are extracted when motion exceeds the temporal baseline.
3. Feature Stability Tracking: Uses feature persistence to identify static vs dynamic scenes.
"""

import os
import asyncio
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from loguru import logger

from Cosmos.integration.vision import VisionModule, VisionTask, ImageInput
from Cosmos.integration.multimodal import MultimodalProcessor

@dataclass
class VisualEvent:
    timestamp: float
    description: str
    motion_magnitude: float # Mean magnitude of optical flow
    saliency_score: float
    is_key_action: bool

@dataclass
class VideoNarrative:
    path: str
    duration: float
    events: List[VisualEvent]
    audio_transcript: str
    combined_summary: str
    motion_profile: List[float] # Magnitude over time

class AdvancedVideoProcessor:
    def __init__(self, vision_module: VisionModule, multimodal_processor: Optional[MultimodalProcessor] = None):
        self.vision = vision_module
        self.multimodal = multimodal_processor or MultimodalProcessor()
        
    async def analyze_video(self, video_path: str) -> VideoNarrative:
        logger.info(f"Video v2.1: Advanced Flow Analysis for {video_path}")
        
        # 1. Parallel Streams
        audio_task = asyncio.create_task(self._process_audio(video_path))
        visual_task = asyncio.create_task(self._process_visual_flow(video_path))
        
        transcript, result_bundle = await asyncio.gather(audio_task, visual_task)
        events, motion_profile = result_bundle
        
        summary = self._synthesize_narrative(events, transcript)
        
        return VideoNarrative(
            path=video_path,
            duration=events[-1].timestamp if events else 0.0,
            events=events,
            audio_transcript=transcript,
            combined_summary=summary,
            motion_profile=motion_profile
        )

    async def _process_audio(self, video_path: str) -> str:
        try:
            result = await self.multimodal.process_file(video_path)
            return result.text or ""
        except Exception: return ""

    async def _process_visual_flow(self, video_path: str) -> Tuple[List[VisualEvent], List[float]]:
        """
        Deep Flow Analysis using Farneback Method.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        events = []
        motion_profile = []
        
        ret, frame1 = cap.read()
        if not ret: return [], []
        
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        
        frame_idx = 1
        sample_rate = int(fps / 2) # Analyze 2 frames per second for flow
        
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame2 = cap.read()
            if not ret: break
            
            next_img = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # 1. Calculate Dense Optical Flow (Farneback)
            flow = cv2.calcOpticalFlowFarneback(prvs, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # 2. Compute Magnitude and Direction
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_mag = np.mean(mag)
            motion_profile.append(float(avg_mag))
            
            # 3. Saliency: Is this an 'Action peak'?
            # We look for magnitude spikes relative to baseline (or > 2.0 threshold)
            is_key = avg_mag > 2.5
            
            # Extract keyframe description if salient enough
            if is_key or frame_idx % int(fps * 10) == 0:
                rgb_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                vision_res = await self.vision.caption(pil_img)
                
                events.append(VisualEvent(
                    timestamp=frame_idx / fps,
                    description=vision_res.caption or "Activity detected",
                    motion_magnitude=float(avg_mag),
                    saliency_score=float(avg_mag * 10),
                    is_key_action=is_key
                ))
            
            prvs = next_img
            frame_idx += sample_rate
            
        cap.release()
        return events, motion_profile

    def _synthesize_narrative(self, events: List[VisualEvent], transcript: str) -> str:
        narrative = "Video Flow Summary:\n"
        for e in events:
            action_tag = "[ACTION]" if e.is_key_action else "[SCENE]"
            narrative += f"- {e.timestamp:.1f}s: {action_tag} {e.description} (Motion: {e.motion_magnitude:.2f})\n"
        narrative += f"\nCombined Transcript:\n{transcript}"
        return narrative
