"""
cosmos DeepSeek OCR2 Integration

Parses health documents (lab results, prescriptions, nutrition labels)
using DeepSeek OCR2 vision model via DeepInfra or local deployment.
"""

import os
import re
import json
import base64
import logging
from pathlib import Path
from datetime import date
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

import httpx

from .models import (
    DocumentType,
    LabResult,
    Prescription,
    NutrientInfo,
    FoodItem,
)

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from OCR processing."""
    success: bool
    document_type: DocumentType
    raw_text: str = ""
    structured_data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "document_type": self.document_type.value,
            "raw_text": self.raw_text,
            "structured_data": self.structured_data,
            "confidence": self.confidence,
            "error": self.error,
        }


class DeepSeekOCRParser:
    """
    DeepSeek OCR2 document parser for health documents.

    Supports:
    - Lab results (blood work, metabolic panels, etc.)
    - Prescriptions
    - Nutrition labels
    - Medical reports

    Uses DeepSeek OCR2 via DeepInfra API or direct DeepSeek API.
    """

    # DeepInfra API endpoint
    DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

    # DeepSeek direct API endpoint
    DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

    # Model configurations
    MODELS = {
        "deepinfra": "deepseek-ai/DeepSeek-V2.5",  # Vision-capable model
        "deepseek": "deepseek-chat",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "deepinfra",
    ):
        """
        Initialize the OCR parser.

        Args:
            api_key: API key (uses env var if not provided)
            provider: "deepinfra" or "deepseek"
        """
        self.provider = provider

        if provider == "deepinfra":
            self.api_key = api_key or os.getenv("DEEPINFRA_API_KEY")
            self.api_url = self.DEEPINFRA_URL
            self.model = self.MODELS["deepinfra"]
        else:
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            self.api_url = self.DEEPSEEK_URL
            self.model = self.MODELS["deepseek"]

        self.client = httpx.AsyncClient(timeout=60.0)

    async def parse_document(
        self,
        image_path: Union[str, Path],
        doc_type: DocumentType,
    ) -> OCRResult:
        """
        Parse a health document image.

        Args:
            image_path: Path to the image file
            doc_type: Type of document to parse

        Returns:
            OCRResult with structured data
        """
        if not self.api_key:
            logger.error("No API key configured for OCR")
            return OCRResult(
                success=False,
                document_type=doc_type,
                error="No API key configured. Set DEEPINFRA_API_KEY or DEEPSEEK_API_KEY.",
            )

        try:
            # Read and encode the image
            image_data = self._encode_image(image_path)
            if not image_data:
                return OCRResult(
                    success=False,
                    document_type=doc_type,
                    error=f"Could not read image: {image_path}",
                )

            # Build the prompt based on document type
            prompt = self._build_prompt(doc_type)

            # Call the vision API
            response = await self._call_vision_api(image_data, prompt)

            if not response:
                return OCRResult(
                    success=False,
                    document_type=doc_type,
                    error="API call failed",
                )

            # Parse the response based on document type
            result = self._parse_response(response, doc_type)

            return result

        except Exception as e:
            logger.error(f"OCR parsing error: {e}")
            return OCRResult(
                success=False,
                document_type=doc_type,
                error=str(e),
            )

    def _encode_image(self, image_path: Union[str, Path]) -> Optional[str]:
        """Encode image to base64."""
        path = Path(image_path)
        if not path.exists():
            logger.error(f"Image not found: {path}")
            return None

        # Determine MIME type
        suffix = path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(suffix, "image/jpeg")

        try:
            with open(path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime_type};base64,{image_data}"
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return None

    def _build_prompt(self, doc_type: DocumentType) -> str:
        """Build the extraction prompt based on document type."""
        prompts = {
            DocumentType.LAB_RESULT: """Analyze this lab result document and extract all test results.

For each test found, provide:
- test_name: Name of the test
- value: Numeric value
- unit: Unit of measurement
- reference_range_low: Lower bound of normal range (if shown)
- reference_range_high: Upper bound of normal range (if shown)
- status: "normal", "low", "high", or "critical" based on the reference range

Return the data as a JSON array of objects with these fields. Be precise with numbers.
If you cannot read a value clearly, note it with confidence < 0.8.

Example output format:
{
  "results": [
    {
      "test_name": "Glucose",
      "value": 95,
      "unit": "mg/dL",
      "reference_range_low": 70,
      "reference_range_high": 100,
      "status": "normal",
      "confidence": 0.95
    }
  ]
}""",

            DocumentType.PRESCRIPTION: """Analyze this prescription document and extract medication information.

Extract:
- medication_name: Full name of the medication
- dosage: Dosage amount and strength
- frequency: How often to take (e.g., "twice daily", "every 8 hours")
- route: How to take it (oral, topical, etc.)
- prescriber: Doctor's name if visible
- refills_remaining: Number of refills if shown
- instructions: Any special instructions
- warnings: Any warnings mentioned

Return the data as JSON:
{
  "medications": [
    {
      "medication_name": "Lisinopril",
      "dosage": "10mg",
      "frequency": "once daily",
      "route": "oral",
      "prescriber": "Dr. Smith",
      "refills_remaining": 3,
      "instructions": "Take in the morning with water",
      "warnings": ["May cause dizziness"],
      "confidence": 0.9
    }
  ]
}""",

            DocumentType.NUTRITION_LABEL: """Analyze this nutrition label and extract all nutritional information.

Extract per serving:
- serving_size: Serving size amount
- serving_unit: Unit (g, ml, oz, etc.)
- calories: Total calories
- protein_g: Protein in grams
- carbs_g: Total carbohydrates in grams
- fat_g: Total fat in grams
- fiber_g: Dietary fiber in grams
- sugar_g: Sugars in grams
- sodium_mg: Sodium in milligrams
- cholesterol_mg: Cholesterol in milligrams
- saturated_fat_g: Saturated fat in grams
- trans_fat_g: Trans fat in grams
- Any vitamins and minerals shown

Return as JSON:
{
  "product_name": "Product name if visible",
  "serving_size": 100,
  "serving_unit": "g",
  "nutrients": {
    "calories": 200,
    "protein_g": 10,
    "carbs_g": 25,
    "fat_g": 8,
    "fiber_g": 3,
    "sugar_g": 5,
    "sodium_mg": 150,
    ...
  },
  "confidence": 0.95
}""",

            DocumentType.MEDICAL_REPORT: """Analyze this medical report and extract key information.

Extract:
- report_type: Type of report (e.g., "Radiology", "Pathology", "Physical Exam")
- date: Date of the report
- provider: Healthcare provider/facility name
- patient_info: Any visible patient information
- findings: List of key findings
- diagnosis: Any diagnoses mentioned
- recommendations: Any recommendations
- follow_up: Follow-up instructions if any

Return as JSON with these fields. Focus on medical findings and recommendations.
{
  "report_type": "...",
  "date": "YYYY-MM-DD",
  "provider": "...",
  "findings": ["..."],
  "diagnosis": ["..."],
  "recommendations": ["..."],
  "confidence": 0.85
}""",
        }

        return prompts.get(doc_type, prompts[DocumentType.LAB_RESULT])

    async def _call_vision_api(
        self,
        image_data: str,
        prompt: str,
    ) -> Optional[str]:
        """Call the vision API with the image."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build the request payload
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data,
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1,  # Low temperature for accuracy
        }

        try:
            response = await self.client.post(
                self.api_url,
                headers=headers,
                json=payload,
            )

            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None

            data = response.json()
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"API call error: {e}")
            return None

    def _parse_response(
        self,
        response: str,
        doc_type: DocumentType,
    ) -> OCRResult:
        """Parse the API response into structured data."""
        try:
            # Extract JSON from the response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return OCRResult(
                    success=False,
                    document_type=doc_type,
                    raw_text=response,
                    error="Could not parse JSON from response",
                )

            data = json.loads(json_match.group())

            # Process based on document type
            if doc_type == DocumentType.LAB_RESULT:
                return self._process_lab_results(data, response)
            elif doc_type == DocumentType.PRESCRIPTION:
                return self._process_prescription(data, response)
            elif doc_type == DocumentType.NUTRITION_LABEL:
                return self._process_nutrition_label(data, response)
            else:
                return OCRResult(
                    success=True,
                    document_type=doc_type,
                    raw_text=response,
                    structured_data=data,
                    confidence=data.get("confidence", 0.8),
                )

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return OCRResult(
                success=False,
                document_type=doc_type,
                raw_text=response,
                error=f"JSON parse error: {e}",
            )

    def _process_lab_results(
        self,
        data: Dict[str, Any],
        raw_text: str,
    ) -> OCRResult:
        """Process lab results into LabResult objects."""
        results = []
        raw_results = data.get("results", [])

        for item in raw_results:
            try:
                lab_result = LabResult(
                    test_name=item.get("test_name", ""),
                    value=float(item.get("value", 0)),
                    unit=item.get("unit", ""),
                    reference_range_low=float(item.get("reference_range_low")) if item.get("reference_range_low") else None,
                    reference_range_high=float(item.get("reference_range_high")) if item.get("reference_range_high") else None,
                    status=item.get("status", "normal"),
                    confidence=float(item.get("confidence", 0.9)),
                )
                results.append(lab_result.to_dict())
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing lab result: {e}")
                continue

        avg_confidence = (
            sum(r.get("confidence", 0.8) for r in results) / len(results)
            if results else 0.0
        )

        return OCRResult(
            success=len(results) > 0,
            document_type=DocumentType.LAB_RESULT,
            raw_text=raw_text,
            structured_data={"results": results},
            confidence=avg_confidence,
        )

    def _process_prescription(
        self,
        data: Dict[str, Any],
        raw_text: str,
    ) -> OCRResult:
        """Process prescription data into Prescription objects."""
        results = []
        medications = data.get("medications", [])

        for item in medications:
            try:
                rx = Prescription(
                    medication_name=item.get("medication_name", ""),
                    dosage=item.get("dosage", ""),
                    frequency=item.get("frequency", ""),
                    route=item.get("route", "oral"),
                    prescriber=item.get("prescriber"),
                    refills_remaining=int(item.get("refills_remaining", 0)),
                    instructions=item.get("instructions", ""),
                    warnings=item.get("warnings", []),
                    confidence=float(item.get("confidence", 0.9)),
                )
                results.append(rx.to_dict())
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing prescription: {e}")
                continue

        avg_confidence = (
            sum(r.get("confidence", 0.8) for r in results) / len(results)
            if results else 0.0
        )

        return OCRResult(
            success=len(results) > 0,
            document_type=DocumentType.PRESCRIPTION,
            raw_text=raw_text,
            structured_data={"medications": results},
            confidence=avg_confidence,
        )

    def _process_nutrition_label(
        self,
        data: Dict[str, Any],
        raw_text: str,
    ) -> OCRResult:
        """Process nutrition label into NutrientInfo."""
        try:
            nutrients = data.get("nutrients", {})

            nutrient_info = NutrientInfo(
                calories=float(nutrients.get("calories", 0)),
                protein_g=float(nutrients.get("protein_g", 0)),
                carbs_g=float(nutrients.get("carbs_g", 0)),
                fat_g=float(nutrients.get("fat_g", 0)),
                fiber_g=float(nutrients.get("fiber_g", 0)),
                sugar_g=float(nutrients.get("sugar_g", 0)),
                sodium_mg=float(nutrients.get("sodium_mg", 0)),
                cholesterol_mg=float(nutrients.get("cholesterol_mg", 0)),
                saturated_fat_g=float(nutrients.get("saturated_fat_g", 0)),
                trans_fat_g=float(nutrients.get("trans_fat_g", 0)),
            )

            food_item = FoodItem(
                name=data.get("product_name", "Scanned Product"),
                serving_size=float(data.get("serving_size", 100)),
                serving_unit=data.get("serving_unit", "g"),
                nutrients=nutrient_info,
            )

            return OCRResult(
                success=True,
                document_type=DocumentType.NUTRITION_LABEL,
                raw_text=raw_text,
                structured_data=food_item.to_dict(),
                confidence=float(data.get("confidence", 0.9)),
            )

        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing nutrition label: {e}")
            return OCRResult(
                success=False,
                document_type=DocumentType.NUTRITION_LABEL,
                raw_text=raw_text,
                error=f"Parse error: {e}",
            )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience function for one-off parsing
async def parse_health_document(
    image_path: Union[str, Path],
    doc_type: str = "lab_result",
    api_key: Optional[str] = None,
) -> OCRResult:
    """
    Parse a health document.

    Args:
        image_path: Path to the image
        doc_type: Type of document (lab_result, prescription, nutrition_label, medical_report)
        api_key: Optional API key

    Returns:
        OCRResult with parsed data
    """
    doc_type_enum = DocumentType(doc_type)

    async with DeepSeekOCRParser(api_key=api_key) as parser:
        return await parser.parse_document(image_path, doc_type_enum)
