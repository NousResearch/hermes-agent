"""
cosmos Vision Module - Image Understanding

Novel Approaches:
1. Multi-Model Ensemble - CLIP + BLIP for robust understanding
2. Visual Question Answering - Answer questions about images
3. Scene Graph Generation - Extract relationships from images
4. Image-to-Memory - Store visual information in memory system
"""

import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Any, Callable, Union
import json
import io

from loguru import logger

# Lazy imports for heavy dependencies
_clip_model = None
_clip_processor = None
_blip_model = None
_blip_processor = None


class VisionTask(Enum):
    """Types of vision tasks."""
    CAPTION = "caption"           # Generate image description
    VQA = "vqa"                   # Visual question answering
    CLASSIFY = "classify"         # Image classification
    DETECT = "detect"             # Object detection
    EMBED = "embed"               # Generate image embedding
    OCR = "ocr"                   # Extract text from image
    SIMILARITY = "similarity"     # Compare images


@dataclass
class ImageInput:
    """Input image for processing."""
    source: Union[str, bytes, Path]  # Path, URL, or raw bytes
    source_type: str = "auto"  # "path", "url", "bytes", "base64", "auto"

    # Metadata
    filename: Optional[str] = None
    mime_type: Optional[str] = None

    # Preprocessing
    max_size: int = 1024  # Max dimension
    normalize: bool = True


@dataclass
class VisionResult:
    """Result from vision processing."""
    task: VisionTask
    success: bool

    # Results by task type
    caption: Optional[str] = None
    answer: Optional[str] = None
    labels: list[dict] = field(default_factory=list)  # [{label, confidence}]
    objects: list[dict] = field(default_factory=list)  # [{label, bbox, confidence}]
    embedding: Optional[list[float]] = None
    text: Optional[str] = None  # OCR result
    similarity_score: Optional[float] = None

    # Metadata
    model_used: str = ""
    processing_time_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "task": self.task.value,
            "success": self.success,
            "caption": self.caption,
            "answer": self.answer,
            "labels": self.labels,
            "objects": self.objects,
            "text": self.text,
            "model": self.model_used,
        }


@dataclass
class SceneGraph:
    """Scene graph extracted from image."""
    objects: list[dict] = field(default_factory=list)  # [{id, label, bbox, attributes}]
    relationships: list[dict] = field(default_factory=list)  # [{subject_id, predicate, object_id}]
    attributes: list[dict] = field(default_factory=list)  # [{object_id, attribute, value}]

    def to_dict(self) -> dict:
        return {
            "objects": self.objects,
            "relationships": self.relationships,
            "attributes": self.attributes,
        }


class VisionModule:
    """
    Multi-model vision understanding system.

    Features:
    - CLIP for embeddings and zero-shot classification
    - BLIP for captioning and VQA
    - Optional OCR with EasyOCR
    - Scene graph generation
    """

    def __init__(
        self,
        device: str = "auto",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        blip_model_name: str = "Salesforce/blip-image-captioning-base",
        use_gpu: bool = True,
    ):
        self.device = self._detect_device(device, use_gpu)
        self.clip_model_name = clip_model_name
        self.blip_model_name = blip_model_name

        self._clip_loaded = False
        self._blip_loaded = False
        self._ocr_loaded = False

        self._lock = asyncio.Lock()

        logger.info(f"Vision module initialized (device: {self.device})")

    def _detect_device(self, device: str, use_gpu: bool) -> str:
        """Detect best available device."""
        if device != "auto":
            return device

        if not use_gpu:
            return "cpu"

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            logger.debug("PyTorch not available for GPU detection, defaulting to CPU")

        return "cpu"

    async def _load_clip(self):
        """Load CLIP model lazily."""
        global _clip_model, _clip_processor

        if self._clip_loaded:
            return

        async with self._lock:
            if self._clip_loaded:
                return

            try:
                from transformers import CLIPProcessor, CLIPModel

                logger.info(f"Loading CLIP model: {self.clip_model_name}")

                loop = asyncio.get_event_loop()
                _clip_processor = await loop.run_in_executor(
                    None,
                    lambda: CLIPProcessor.from_pretrained(self.clip_model_name)
                )
                _clip_model = await loop.run_in_executor(
                    None,
                    lambda: CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
                )

                self._clip_loaded = True
                logger.info("CLIP model loaded")

            except Exception as e:
                logger.error(f"Failed to load CLIP: {e}")
                raise

    async def _load_blip(self):
        """Load BLIP model lazily."""
        global _blip_model, _blip_processor

        if self._blip_loaded:
            return

        async with self._lock:
            if self._blip_loaded:
                return

            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration

                logger.info(f"Loading BLIP model: {self.blip_model_name}")

                loop = asyncio.get_event_loop()
                _blip_processor = await loop.run_in_executor(
                    None,
                    lambda: BlipProcessor.from_pretrained(self.blip_model_name)
                )
                _blip_model = await loop.run_in_executor(
                    None,
                    lambda: BlipForConditionalGeneration.from_pretrained(
                        self.blip_model_name
                    ).to(self.device)
                )

                self._blip_loaded = True
                logger.info("BLIP model loaded")

            except Exception as e:
                logger.error(f"Failed to load BLIP: {e}")
                raise

    async def _load_image(self, input: ImageInput) -> "PIL.Image.Image":
        """Load image from various sources."""
        from PIL import Image

        loop = asyncio.get_event_loop()

        if input.source_type == "auto":
            if isinstance(input.source, bytes):
                input.source_type = "bytes"
            elif isinstance(input.source, Path) or (
                isinstance(input.source, str) and not input.source.startswith(('http://', 'https://'))
            ):
                input.source_type = "path"
            elif isinstance(input.source, str) and input.source.startswith(('http://', 'https://')):
                input.source_type = "url"
            elif isinstance(input.source, str) and len(input.source) > 200:
                input.source_type = "base64"

        if input.source_type == "path":
            image = await loop.run_in_executor(
                None, lambda: Image.open(input.source).convert("RGB")
            )

        elif input.source_type == "url":
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(input.source) as response:
                    data = await response.read()
            image = Image.open(io.BytesIO(data)).convert("RGB")

        elif input.source_type == "bytes":
            image = Image.open(io.BytesIO(input.source)).convert("RGB")

        elif input.source_type == "base64":
            data = base64.b64decode(input.source)
            image = Image.open(io.BytesIO(data)).convert("RGB")

        else:
            raise ValueError(f"Unknown source type: {input.source_type}")

        # Resize if needed
        if input.max_size:
            max_dim = max(image.size)
            if max_dim > input.max_size:
                scale = input.max_size / max_dim
                new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
                image = image.resize(new_size, Image.LANCZOS)

        return image

    async def caption(
        self,
        image: Union[ImageInput, str, Path],
        max_length: int = 50,
        num_beams: int = 4,
    ) -> VisionResult:
        """
        Generate a caption for an image.

        Uses BLIP for high-quality captions.
        """
        import time
        start = time.time()

        if not isinstance(image, ImageInput):
            image = ImageInput(source=image)

        try:
            await self._load_blip()

            pil_image = await self._load_image(image)

            # Process with BLIP
            loop = asyncio.get_event_loop()

            inputs = await loop.run_in_executor(
                None,
                lambda: _blip_processor(pil_image, return_tensors="pt").to(self.device)
            )

            output_ids = await loop.run_in_executor(
                None,
                lambda: _blip_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                )
            )

            caption = _blip_processor.decode(output_ids[0], skip_special_tokens=True)

            return VisionResult(
                task=VisionTask.CAPTION,
                success=True,
                caption=caption,
                model_used=self.blip_model_name,
                processing_time_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            logger.error(f"Captioning failed: {e}")
            return VisionResult(
                task=VisionTask.CAPTION,
                success=False,
                error=str(e),
            )

    async def answer_question(
        self,
        image: Union[ImageInput, str, Path],
        question: str,
        max_length: int = 50,
    ) -> VisionResult:
        """
        Answer a question about an image (VQA).

        Uses BLIP for visual question answering.
        """
        import time
        start = time.time()

        if not isinstance(image, ImageInput):
            image = ImageInput(source=image)

        try:
            await self._load_blip()

            pil_image = await self._load_image(image)

            loop = asyncio.get_event_loop()

            # Use BLIP for VQA
            inputs = await loop.run_in_executor(
                None,
                lambda: _blip_processor(pil_image, question, return_tensors="pt").to(self.device)
            )

            output_ids = await loop.run_in_executor(
                None,
                lambda: _blip_model.generate(**inputs, max_length=max_length)
            )

            answer = _blip_processor.decode(output_ids[0], skip_special_tokens=True)

            return VisionResult(
                task=VisionTask.VQA,
                success=True,
                answer=answer,
                model_used=self.blip_model_name,
                processing_time_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            logger.error(f"VQA failed: {e}")
            return VisionResult(
                task=VisionTask.VQA,
                success=False,
                error=str(e),
            )

    async def classify(
        self,
        image: Union[ImageInput, str, Path],
        labels: list[str],
        top_k: int = 5,
    ) -> VisionResult:
        """
        Zero-shot image classification.

        Uses CLIP for flexible classification with any labels.
        """
        import time
        start = time.time()

        if not isinstance(image, ImageInput):
            image = ImageInput(source=image)

        try:
            await self._load_clip()

            import torch

            pil_image = await self._load_image(image)

            loop = asyncio.get_event_loop()

            # Process image and text
            inputs = await loop.run_in_executor(
                None,
                lambda: _clip_processor(
                    text=labels,
                    images=pil_image,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
            )

            outputs = await loop.run_in_executor(
                None,
                lambda: _clip_model(**inputs)
            )

            # Get probabilities
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

            # Get top-k results
            top_probs, top_indices = probs[0].topk(min(top_k, len(labels)))

            results = [
                {"label": labels[idx], "confidence": float(prob)}
                for prob, idx in zip(top_probs.cpu(), top_indices.cpu())
            ]

            return VisionResult(
                task=VisionTask.CLASSIFY,
                success=True,
                labels=results,
                model_used=self.clip_model_name,
                processing_time_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return VisionResult(
                task=VisionTask.CLASSIFY,
                success=False,
                error=str(e),
            )

    async def embed(
        self,
        image: Union[ImageInput, str, Path],
    ) -> VisionResult:
        """
        Generate image embedding using CLIP.

        Useful for similarity search and retrieval.
        """
        import time
        start = time.time()

        if not isinstance(image, ImageInput):
            image = ImageInput(source=image)

        try:
            await self._load_clip()

            pil_image = await self._load_image(image)

            loop = asyncio.get_event_loop()

            inputs = await loop.run_in_executor(
                None,
                lambda: _clip_processor(images=pil_image, return_tensors="pt").to(self.device)
            )

            outputs = await loop.run_in_executor(
                None,
                lambda: _clip_model.get_image_features(**inputs)
            )

            # Normalize embedding
            embedding = outputs[0].cpu().detach().numpy()
            embedding = embedding / (embedding ** 2).sum() ** 0.5

            return VisionResult(
                task=VisionTask.EMBED,
                success=True,
                embedding=embedding.tolist(),
                model_used=self.clip_model_name,
                processing_time_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return VisionResult(
                task=VisionTask.EMBED,
                success=False,
                error=str(e),
            )

    async def extract_text(
        self,
        image: Union[ImageInput, str, Path],
        languages: list[str] = ["en"],
    ) -> VisionResult:
        """
        Extract text from image using OCR.

        Uses EasyOCR for multi-language support.
        """
        import time
        start = time.time()

        if not isinstance(image, ImageInput):
            image = ImageInput(source=image)

        try:
            import easyocr

            pil_image = await self._load_image(image)

            loop = asyncio.get_event_loop()

            # Initialize reader
            reader = await loop.run_in_executor(
                None,
                lambda: easyocr.Reader(languages, gpu=self.device != "cpu")
            )

            # Convert PIL to numpy
            import numpy as np
            image_array = np.array(pil_image)

            # Run OCR
            results = await loop.run_in_executor(
                None,
                lambda: reader.readtext(image_array)
            )

            # Extract text
            text_parts = [r[1] for r in results]
            full_text = " ".join(text_parts)

            return VisionResult(
                task=VisionTask.OCR,
                success=True,
                text=full_text,
                model_used="easyocr",
                processing_time_ms=(time.time() - start) * 1000,
            )

        except ImportError:
            logger.warning("EasyOCR not installed. Install with: pip install easyocr")
            return VisionResult(
                task=VisionTask.OCR,
                success=False,
                error="EasyOCR not installed",
            )
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return VisionResult(
                task=VisionTask.OCR,
                success=False,
                error=str(e),
            )

    async def compare_images(
        self,
        image1: Union[ImageInput, str, Path],
        image2: Union[ImageInput, str, Path],
    ) -> VisionResult:
        """
        Compare two images using CLIP embeddings.

        Returns similarity score between 0 and 1.
        """
        import time
        start = time.time()

        try:
            # Get embeddings for both images
            result1 = await self.embed(image1)
            result2 = await self.embed(image2)

            if not result1.success or not result2.success:
                return VisionResult(
                    task=VisionTask.SIMILARITY,
                    success=False,
                    error="Failed to embed one or both images",
                )

            # Compute cosine similarity
            import numpy as np
            emb1 = np.array(result1.embedding)
            emb2 = np.array(result2.embedding)

            similarity = float(np.dot(emb1, emb2))

            return VisionResult(
                task=VisionTask.SIMILARITY,
                success=True,
                similarity_score=similarity,
                model_used=self.clip_model_name,
                processing_time_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return VisionResult(
                task=VisionTask.SIMILARITY,
                success=False,
                error=str(e),
            )

    async def analyze(
        self,
        image: Union[ImageInput, str, Path],
        include_caption: bool = True,
        include_labels: bool = True,
        include_ocr: bool = False,
        labels: Optional[list[str]] = None,
    ) -> dict:
        """
        Comprehensive image analysis.

        Runs multiple vision tasks and combines results.
        """
        if not isinstance(image, ImageInput):
            image = ImageInput(source=image)

        results = {}

        tasks = []

        if include_caption:
            tasks.append(("caption", self.caption(image)))

        if include_labels:
            default_labels = [
                "person", "animal", "vehicle", "building", "nature",
                "food", "text", "art", "technology", "indoor", "outdoor"
            ]
            tasks.append(("classify", self.classify(image, labels or default_labels)))

        if include_ocr:
            tasks.append(("ocr", self.extract_text(image)))

        # Run tasks concurrently
        for name, task in tasks:
            result = await task
            results[name] = result.to_dict()

        return results

    async def generate_scene_graph(
        self,
        image: Union[ImageInput, str, Path],
        llm_fn: Optional[Callable] = None,
    ) -> SceneGraph:
        """
        Generate a scene graph from an image.

        Uses caption + LLM to extract structured relationships.
        """
        if not isinstance(image, ImageInput):
            image = ImageInput(source=image)

        # Get caption first
        caption_result = await self.caption(image)

        if not caption_result.success:
            return SceneGraph()

        if not llm_fn:
            # Basic parsing without LLM
            return self._parse_caption_to_graph(caption_result.caption)

        # Use LLM to extract scene graph
        prompt = f"""Extract a scene graph from this image description.

Description: {caption_result.caption}

Return JSON with:
- objects: [{{"id": 1, "label": "object name", "attributes": ["attr1"]}}]
- relationships: [{{"subject_id": 1, "predicate": "on/near/holding/etc", "object_id": 2}}]

Example:
{{"objects": [{{"id": 1, "label": "cat", "attributes": ["orange", "sitting"]}}, {{"id": 2, "label": "couch", "attributes": ["gray"]}}], "relationships": [{{"subject_id": 1, "predicate": "on", "object_id": 2}}]}}"""

        try:
            if asyncio.iscoroutinefunction(llm_fn):
                response = await llm_fn(prompt)
            else:
                response = llm_fn(prompt)

            # Extract JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])

                return SceneGraph(
                    objects=data.get("objects", []),
                    relationships=data.get("relationships", []),
                )

        except Exception as e:
            logger.error(f"Scene graph extraction failed: {e}")

        return self._parse_caption_to_graph(caption_result.caption)

    def _parse_caption_to_graph(self, caption: str) -> SceneGraph:
        """Basic caption to scene graph parsing."""
        import re

        graph = SceneGraph()

        # Simple noun extraction
        words = caption.lower().split()
        objects = []

        # Common objects
        object_words = {"person", "man", "woman", "child", "dog", "cat", "car",
                       "tree", "building", "table", "chair", "phone", "computer"}

        for i, word in enumerate(words):
            if word in object_words:
                objects.append({
                    "id": len(graph.objects) + 1,
                    "label": word,
                    "attributes": [],
                })

        graph.objects = objects
        return graph

    def get_stats(self) -> dict:
        """Get vision module statistics."""
        return {
            "device": self.device,
            "clip_loaded": self._clip_loaded,
            "blip_loaded": self._blip_loaded,
            "clip_model": self.clip_model_name,
            "blip_model": self.blip_model_name,
        }
