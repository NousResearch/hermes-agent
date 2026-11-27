"""Monet Project Primer protocol and package builder.

A Project Primer is a secret-free, agent-generated project configuration. It
travels inside the existing ``.monetproj`` ZIP format as ``primer.json`` so
older Monet releases safely ignore it while v4 clients can present a resumable
connector checklist.

The four non-negotiable protocol rules are:

1. Primer files contain configuration and secret *slot descriptions*, never
   credential values, cookies, private keys, or executable setup scripts.
2. The existing formatVersion=1 package remains additive and compatible.
3. iPad is the recommended review surface; macOS and Windows are supported
   desktop alternatives. Linux is deliberately unsupported.
4. Opening Monet or an installer is always an explicit user choice.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import re
import stat
import struct
import subprocess
import sys
import zipfile
import zlib
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import parse_qsl, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

try:
    from .project_package import CONNECTOR_KINDS, FORMAT_VERSION
except ImportError:  # Standalone copy bundled with the Hermes skill.
    FORMAT_VERSION = 1
    CONNECTOR_KINDS = {
        "gdrive",
        "dropbox",
        "icloud",
        "files",
        "mcp",
        "vercel",
        "render",
        "supabase",
        "directus",
        "github",
        "wix",
        "linear",
        "slack",
        "posthog",
        "postmark",
        "google-cloud",
        "google-merchant",
        "other",
    }

PRIMER_KIND = "iammonet.project-primer"
PRIMER_SCHEMA_VERSION = 1
PRIMER_SETUP_URL = "https://iammonet.com/setup"
MAX_PRIMER_JSON_BYTES = 256_000
MAX_INLINE_SETUP_URL_CHARS = 2_900
MAX_PREVIEW_PAGES = 100
MAX_PREVIEW_IMAGE_BYTES = 64_000_000
MAX_PREVIEW_TOTAL_BYTES = 500_000_000
MAX_PREVIEW_IMAGE_PIXELS = 100_000_000
AGENT_PREVIEW_CAPTURE_KIND = "agent-preview"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{1,127}$")
_SECRET_KEY = re.compile(
    r"(?i)(?:^|[_-])(password|passwd|secret|token|api[_-]?key|private[_-]?key|cookies?|authorization|credentials?)(?:$|[_-])"
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----"),
    re.compile(r"\bgh[opusr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/-]{20,}={0,2}\b"),
)


class PrimerError(ValueError):
    """Raised when a Primer is unsafe or cannot be packaged."""


def _safe_validation_message(label: str, error: ValidationError) -> str:
    """Describe validation failures without echoing attacker-controlled input."""

    details: list[str] = []
    for item in error.errors(include_url=False, include_input=False)[:8]:
        location = ".".join(str(component) for component in item.get("loc", ()))
        message = str(item.get("msg", "invalid value"))
        details.append(f"{location}: {message}" if location else message)
    suffix = "; ".join(details) if details else "validation failed"
    return f"Invalid {label}: {suffix}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ProjectKind(StrEnum):
    WEBSITE = "website"
    PWA = "pwa"
    APP = "app"


class CrawlSource(StrEnum):
    LIVE = "live"
    DEV = "dev"
    LOCAL = "local"


class Viewport(StrEnum):
    DESKTOP = "desktop"
    MOBILE = "mobile"


class ColorScheme(StrEnum):
    LIGHT = "light"
    DARK = "dark"


class CaptureBrowser(StrEnum):
    CHROMIUM = "chromium"
    CHROME = "chrome"
    EDGE = "edge"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


class RenderingMode(StrEnum):
    STATIC = "static"
    SPA = "spa"
    SSR = "ssr"
    HYBRID = "hybrid"
    NATIVE = "native"
    UNKNOWN = "unknown"


class AuthMode(StrEnum):
    NONE = "none"
    KEYCHAIN = "keychain"
    OAUTH = "oauth"
    INTERACTIVE_SESSION = "interactive-session"


class RecommendedSurface(StrEnum):
    IPAD = "ipad"
    DESKTOP = "desktop"


class GeneratedBy(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    version: str | None = Field(default=None, max_length=80)


class RepositoryContext(StrictModel):
    provider: Literal["github", "gitlab", "bitbucket", "local", "other"] = "github"
    repository: str | None = Field(default=None, max_length=240)
    branch: str | None = Field(default=None, max_length=160)
    subdirectory: str | None = Field(default=None, max_length=500)
    design_paths: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("design_paths")
    @classmethod
    def validate_design_paths(cls, paths: list[str]) -> list[str]:
        for path in paths:
            if (
                not path
                or len(path) > 1_000
                or path.startswith("/")
                or "\0" in path
                or ".." in path.split("/")
            ):
                raise ValueError("design paths must be repository-relative paths without traversal")
        return paths


class PrimerProject(StrictModel):
    slug: str
    name: str = Field(min_length=1, max_length=200)
    kind: ProjectKind = ProjectKind.WEBSITE
    live_url: str | None = None
    dev_url: str | None = None
    local_url: str | None = None
    description: str | None = Field(default=None, max_length=4_000)
    known_facts: str | None = Field(default=None, max_length=50_000)
    design_markdown: str | None = Field(default=None, max_length=200_000)
    repository: RepositoryContext | None = None

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, value: str) -> str:
        if not _SLUG.fullmatch(value):
            raise ValueError("must be a lowercase URL-safe slug (letters, numbers, hyphens)")
        return value

    @field_validator("live_url", "dev_url", "local_url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_http_url(value)

    @model_validator(mode="after")
    def require_capture_url(self) -> PrimerProject:
        if self.kind != ProjectKind.APP and not any((self.live_url, self.dev_url, self.local_url)):
            raise ValueError("website and PWA Primers need at least one capture URL")
        return self


class TechnologyStack(StrictModel):
    frameworks: list[str] = Field(default_factory=list, max_length=30)
    languages: list[str] = Field(default_factory=list, max_length=30)
    package_manager: str | None = Field(default=None, max_length=80)
    rendering: RenderingMode = RenderingMode.UNKNOWN
    hosting: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("frameworks", "languages")
    @classmethod
    def validate_stack_items(cls, values: list[str]) -> list[str]:
        if any(not value or len(value) > 120 for value in values):
            raise ValueError("stack entries must contain 1 to 120 characters")
        return values

    @field_validator("hosting")
    @classmethod
    def validate_hosting_items(cls, values: list[str]) -> list[str]:
        if any(not value or len(value) > 160 for value in values):
            raise ValueError("hosting entries must contain 1 to 160 characters")
        return values


class CaptureProfile(StrictModel):
    preferred_source: CrawlSource = CrawlSource.LIVE
    viewport: Viewport = Viewport.DESKTOP
    color_scheme: ColorScheme = ColorScheme.LIGHT
    max_depth: Annotated[int, Field(ge=0, le=5)] = 2
    max_pages: Annotated[int, Field(ge=1, le=100)] = 50
    extra_wait_ms: Annotated[int, Field(ge=0, le=30_000)] = 0
    template_page_cap: Annotated[int, Field(ge=0, le=100)] = 3
    open_menus: bool = False
    wait_for_fonts: bool = True
    wait_for_dom_stability: bool = True
    additional_paths: list[str] = Field(default_factory=list, max_length=100)
    include_patterns: list[str] = Field(default_factory=list, max_length=100)
    exclude_patterns: list[str] = Field(default_factory=list, max_length=100)
    representative_url: str | None = None

    @field_validator("representative_url")
    @classmethod
    def validate_representative_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_http_url(value)

    @field_validator("additional_paths")
    @classmethod
    def validate_paths(cls, paths: list[str]) -> list[str]:
        for path in paths:
            if (
                not path
                or len(path) > 2_000
                or not path.startswith("/")
                or "\0" in path
                or ".." in path.split("/")
            ):
                raise ValueError("additional paths must be absolute URL paths without traversal")
        return paths

    @field_validator("include_patterns", "exclude_patterns")
    @classmethod
    def validate_patterns(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            if not pattern or len(pattern) > 500 or not pattern.startswith("/") or "\0" in pattern:
                raise ValueError("crawl patterns must be absolute path globs")
        return patterns


class AgentPreviewPage(StrictModel):
    """One public rendered page included in an Agent Preview Pack."""

    url: str
    page_slug: str
    screenshot: str
    page_title: str | None = Field(default=None, max_length=500)
    viewport_width: Annotated[int | None, Field(default=None, ge=1, le=10_000)]
    viewport_height: Annotated[int | None, Field(default=None, ge=1, le=100_000)]
    scroll_height: Annotated[int | None, Field(default=None, ge=1, le=250_000)]
    meta_description: str | None = Field(default=None, max_length=5_000)
    canonical_url: str | None = None
    h1: str | None = Field(default=None, max_length=2_000)

    @field_validator("url", "canonical_url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_http_url(value)

    @field_validator("page_slug")
    @classmethod
    def validate_page_slug(cls, value: str) -> str:
        if not _SLUG.fullmatch(value):
            raise ValueError("page_slug must be a lowercase URL-safe path segment")
        return value

    @field_validator("screenshot")
    @classmethod
    def validate_screenshot_path(cls, value: str) -> str:
        path = Path(value)
        if (
            not value
            or len(value) > 1_000
            or path.is_absolute()
            or "\0" in value
            or ".." in path.parts
            or path.suffix.lower() != ".png"
        ):
            raise ValueError("screenshot must be a relative PNG path without traversal")
        return path.as_posix()


class AgentPreview(StrictModel):
    """Secret-free metadata for one agent-rendered review baseline."""

    label: str = Field(default="Hermes Preview", min_length=1, max_length=160)
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    captured_with: CaptureBrowser = CaptureBrowser.CHROMIUM
    viewport: Viewport = Viewport.DESKTOP
    color_scheme: ColorScheme = ColorScheme.LIGHT
    notes: str | None = Field(default=None, max_length=4_000)
    pages: list[AgentPreviewPage] = Field(min_length=1, max_length=MAX_PREVIEW_PAGES)

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if any(character in value for character in ("/", "\\", "\0")):
            raise ValueError("preview label must be one safe path component")
        return value

    @model_validator(mode="after")
    def validate_preview(self) -> AgentPreview:
        slugs = [page.page_slug for page in self.pages]
        if len(slugs) != len(set(slugs)):
            raise ValueError("preview page_slug values must be unique")
        _reject_secret_material(
            {
                "label": self.label,
                "notes": self.notes,
                "pages": [
                    page.model_dump(mode="json", exclude={"screenshot"}) for page in self.pages
                ],
            }
        )
        return self


class ResourceReference(StrictModel):
    key: str
    value: str = Field(min_length=1, max_length=2_000)
    label: str | None = Field(default=None, max_length=120)

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("must be a stable lowercase identifier")
        if _SECRET_KEY.search(value):
            raise ValueError("resource references cannot be credential fields")
        return value


class SecretSlot(StrictModel):
    id: str
    label: str = Field(min_length=1, max_length=120)
    purpose: str = Field(min_length=1, max_length=500)
    source_environment_variable: str | None = None
    required: bool = True

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("must be a stable lowercase identifier")
        return value

    @field_validator("source_environment_variable")
    @classmethod
    def validate_env_name(cls, value: str | None) -> str | None:
        if value is not None and not _ENV_NAME.fullmatch(value):
            raise ValueError("must be an environment variable name, not a value")
        return value


class ConnectorAuth(StrictModel):
    mode: AuthMode = AuthMode.NONE
    scopes: list[str] = Field(default_factory=list, max_length=50)
    secret_slots: list[SecretSlot] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_slots(self) -> ConnectorAuth:
        if self.mode == AuthMode.NONE and self.secret_slots:
            raise ValueError("auth mode 'none' cannot declare secret slots")
        return self


class ConnectorValidation(StrictModel):
    kind: Literal[
        "none",
        "url",
        "github-repository",
        "vercel-deployment",
        "files-folder",
        "mcp-handshake",
    ] = "none"
    target: str | None = Field(default=None, max_length=2_000)


class ConnectorRequirement(StrictModel):
    id: str
    kind: str
    name: str = Field(min_length=1, max_length=160)
    required: bool = True
    purpose: str = Field(min_length=1, max_length=1_000)
    url: str | None = None
    resources: list[ResourceReference] = Field(default_factory=list, max_length=50)
    auth: ConnectorAuth = Field(default_factory=ConnectorAuth)
    validation: ConnectorValidation = Field(default_factory=ConnectorValidation)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("must be a stable lowercase identifier")
        return value

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        if value not in CONNECTOR_KINDS:
            raise ValueError("unsupported connector kind")
        return value

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_http_url(value)


class SetupPreferences(StrictModel):
    recommended_surface: RecommendedSurface = RecommendedSurface.IPAD
    offer_desktop_when_local: bool = True
    notes: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, notes: list[str]) -> list[str]:
        if any(not note or len(note) > 1_000 for note in notes):
            raise ValueError("setup notes must contain 1 to 1000 characters")
        return notes


class ProjectPrimer(StrictModel):
    kind: Literal["iammonet.project-primer"] = PRIMER_KIND
    schema_version: Literal[1] = PRIMER_SCHEMA_VERSION
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    generated_by: GeneratedBy
    project: PrimerProject
    stack: TechnologyStack = Field(default_factory=TechnologyStack)
    capture: CaptureProfile = Field(default_factory=CaptureProfile)
    connectors: list[ConnectorRequirement] = Field(default_factory=list, max_length=50)
    setup: SetupPreferences = Field(default_factory=SetupPreferences)

    @model_validator(mode="after")
    def validate_primer(self) -> ProjectPrimer:
        connector_ids = [connector.id for connector in self.connectors]
        if len(connector_ids) != len(set(connector_ids)):
            raise ValueError("connector requirement IDs must be unique")
        slot_ids = [
            slot.id for connector in self.connectors for slot in connector.auth.secret_slots
        ]
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("secret slot IDs must be unique across the Primer")
        for connector in self.connectors:
            resource_keys = [resource.key for resource in connector.resources]
            if len(resource_keys) != len(set(resource_keys)):
                raise ValueError("connector resource keys must be unique")

        source_url = {
            CrawlSource.LIVE: self.project.live_url,
            CrawlSource.DEV: self.project.dev_url,
            CrawlSource.LOCAL: self.project.local_url,
        }[self.capture.preferred_source]
        if self.project.kind != ProjectKind.APP and not source_url:
            raise ValueError("preferred capture source must have a configured URL")

        _reject_secret_material(self.model_dump(mode="json", by_alias=False))
        return self

    def canonical_json(self) -> bytes:
        body = self.model_dump_json(indent=2, by_alias=False).encode("utf-8")
        if len(body) > MAX_PRIMER_JSON_BYTES:
            raise PrimerError("Project Primer is too large.")
        return body


def _validate_http_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("URLs must not contain embedded credentials")
    for key, query_value in parse_qsl(parsed.query, keep_blank_values=True):
        if query_value and _SECRET_KEY.search(key):
            raise ValueError("URL query strings must not contain credential fields")
    return value


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _reject_secret_material(
    value: Any,
    *,
    path: tuple[str, ...] = (),
    allow_slot_metadata: bool = True,
) -> None:
    """Reject credential-looking material anywhere in a Primer.

    Secret slot metadata is allowed in schema-validated Primer data. Values
    are not. Key names are matched on snake_case, kebab-case, and camelCase
    boundaries; a secret-named key whose value is null carries nothing and is
    allowed. Strict Pydantic models already reject unknown ``token``/
    ``password`` fields; this second layer catches credentials pasted into
    notes, URLs, resource references, or other nominally safe strings.
    """

    if isinstance(value, dict):
        for key, child in value.items():
            # These schema fields describe secret *requirements* and are safe.
            if (
                child is not None
                and _SECRET_KEY.search(_CAMEL_BOUNDARY.sub("_", str(key)))
                and not (
                    allow_slot_metadata
                    and key in {"secret_slots", "source_environment_variable"}
                )
            ):
                location = ".".join((*path, str(key)))
                raise ValueError(f"credential field is not allowed at {location}")
            _reject_secret_material(
                child, path=(*path, str(key)), allow_slot_metadata=allow_slot_metadata
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_material(
                child, path=(*path, str(index)), allow_slot_metadata=allow_slot_metadata
            )
        return
    if not isinstance(value, str):
        return
    for pattern in _SECRET_VALUE_PATTERNS:
        if pattern.search(value):
            location = ".".join(path) or "Primer"
            raise ValueError(f"credential-like value is not allowed at {location}")


def reject_secret_material(
    value: Any, *, label: str, allow_slot_metadata: bool = False
) -> None:
    """Reject credential-like keys or values anywhere in parsed JSON data.

    Package members other than ``primer.json`` bypass Pydantic validation, so
    verifiers must run every parsed member through this scan before reporting
    ``containsSecrets: false``. Pass ``allow_slot_metadata=True`` only for
    data derived from a schema-validated Primer, where ``secret_slots``
    describes requirements rather than carrying values.
    """

    _reject_secret_material(value, path=(label,), allow_slot_metadata=allow_slot_metadata)


def load_project_primer(value: str | bytes | Path | dict[str, Any]) -> ProjectPrimer:
    """Load and strictly validate a Primer from JSON, a path, or a mapping."""

    try:
        if isinstance(value, Path):
            metadata = value.stat()
            if value.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise PrimerError("Project Primer must be a regular file.")
            if metadata.st_size > MAX_PRIMER_JSON_BYTES:
                raise PrimerError("Project Primer is too large.")
            with value.open("rb") as stream:
                body = stream.read(MAX_PRIMER_JSON_BYTES + 1)
            if len(body) > MAX_PRIMER_JSON_BYTES:
                raise PrimerError("Project Primer is too large.")
            return ProjectPrimer.model_validate_json(body)
        if isinstance(value, bytes):
            if len(value) > MAX_PRIMER_JSON_BYTES:
                raise PrimerError("Project Primer is too large.")
            return ProjectPrimer.model_validate_json(value)
        if isinstance(value, str):
            body = value.encode("utf-8")
            if len(body) > MAX_PRIMER_JSON_BYTES:
                raise PrimerError("Project Primer is too large.")
            return ProjectPrimer.model_validate_json(body)
        return ProjectPrimer.model_validate(value)
    except PrimerError:
        raise
    except ValidationError as error:
        raise PrimerError(_safe_validation_message("Project Primer", error)) from error
    except (OSError, ValueError) as error:
        raise PrimerError("Invalid Project Primer.") from error


def load_agent_preview(path: Path) -> AgentPreview:
    """Load a strict Agent Preview manifest from a regular JSON file."""

    try:
        metadata = path.stat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise PrimerError("Agent Preview manifest must be a regular file.")
        if metadata.st_size > MAX_PRIMER_JSON_BYTES:
            raise PrimerError("Agent Preview manifest is too large.")
        return AgentPreview.model_validate_json(path.read_bytes())
    except PrimerError:
        raise
    except ValidationError as error:
        raise PrimerError(_safe_validation_message("Agent Preview", error)) from error
    except (OSError, ValueError) as error:
        raise PrimerError("Invalid Agent Preview.") from error


def _configured_project_hosts(primer: ProjectPrimer) -> set[str]:
    hosts: set[str] = set()
    for value in (primer.project.live_url, primer.project.dev_url, primer.project.local_url):
        if value and (hostname := urlparse(value).hostname):
            hosts.add(hostname.lower().removeprefix("www."))
    return hosts


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != _PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise PrimerError(f"Agent Preview screenshot is not a valid PNG: {path.name}")
    width, height = struct.unpack(">II", header[16:24])
    if width == 0 or height == 0 or width * height > MAX_PREVIEW_IMAGE_PIXELS:
        raise PrimerError(f"Agent Preview screenshot dimensions are unsafe: {path.name}")
    return width, height


def _validated_preview_assets(
    primer: ProjectPrimer,
    preview: AgentPreview,
    preview_root: Path,
) -> list[tuple[AgentPreviewPage, Path, int, int]]:
    if len(preview.pages) > primer.capture.max_pages:
        raise PrimerError(
            "Agent Preview contains more pages than the Primer capture.max_pages policy."
        )

    root = preview_root.resolve(strict=True)
    configured_hosts = _configured_project_hosts(primer)
    assets: list[tuple[AgentPreviewPage, Path, int, int]] = []
    total_bytes = 0
    for page in preview.pages:
        hostname = (urlparse(page.url).hostname or "").lower().removeprefix("www.")
        if configured_hosts and hostname not in configured_hosts:
            raise PrimerError(
                f"Agent Preview URL host is not configured in the Primer: {page.url}"
            )

        candidate = preview_root / page.screenshot
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as error:
            raise PrimerError(
                f"Agent Preview screenshot is outside the preview folder: {page.screenshot}"
            ) from error
        relative = candidate.relative_to(preview_root)
        cursor = preview_root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise PrimerError(
                    f"Agent Preview screenshots cannot use symbolic links: {page.screenshot}"
                )
        metadata = resolved.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_PREVIEW_IMAGE_BYTES:
            raise PrimerError(f"Agent Preview screenshot is too large: {page.screenshot}")
        total_bytes += metadata.st_size
        if total_bytes > MAX_PREVIEW_TOTAL_BYTES:
            raise PrimerError("Agent Preview screenshots exceed the total safe size limit.")
        width, height = _png_dimensions(resolved)
        assets.append((page, resolved, width, height))
    return assets


def _connector_config(requirement: ConnectorRequirement) -> dict[str, Any]:
    return {
        "monetPrimerRequirement": {
            "id": requirement.id,
            "required": requirement.required,
            "purpose": requirement.purpose,
            "resources": [resource.model_dump(mode="json") for resource in requirement.resources],
            "auth": requirement.auth.model_dump(mode="json"),
            "validation": requirement.validation.model_dump(mode="json"),
        }
    }


def _b64_json(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(body).decode("ascii")


def _project_dto(primer: ProjectPrimer) -> dict[str, Any]:
    project = primer.project
    repository = project.repository
    sources: dict[str, Any] = {
        "projectPrimer": {
            "schemaVersion": primer.schema_version,
            "generatedBy": primer.generated_by.model_dump(mode="json"),
        }
    }
    if repository:
        sources[repository.provider] = {
            "repository": repository.repository,
            "branch": repository.branch,
            "subdirectory": repository.subdirectory,
            "designPaths": repository.design_paths,
        }

    base_url = project.live_url or project.dev_url or project.local_url
    if project.kind == ProjectKind.APP and not base_url:
        base_url = f"iammonet://app/{project.slug}"

    connectors = []
    for requirement in primer.connectors:
        notes = requirement.purpose
        if requirement.auth.secret_slots:
            slot_labels = ", ".join(slot.label for slot in requirement.auth.secret_slots)
            notes += f"\n\nFinish setup on this device: {slot_labels}."
        connectors.append(
            {
                "kind": requirement.kind,
                "name": requirement.name,
                "url": requirement.url,
                "notes": notes,
                "configJSON": _b64_json(_connector_config(requirement)),
                "driveRef": next(
                    (
                        resource.value
                        for resource in requirement.resources
                        if resource.key in {"drive-file-id", "drive-folder-id"}
                    ),
                    None,
                ),
                "secretRef": None,
            }
        )

    return {
        "slug": project.slug,
        "name": project.name,
        "baseURL": base_url,
        "devURL": project.dev_url,
        "localURL": project.local_url,
        "repoPath": (
            repository.repository if repository and repository.provider == "local" else None
        ),
        "descriptionText": project.description,
        "designMd": project.design_markdown,
        "knownFacts": project.known_facts,
        "sourcesJSON": _b64_json(sources),
        "projectKind": project.kind.value,
        "userPreferredSource": primer.capture.preferred_source.value,
        "crawlExtraWaitMs": primer.capture.extra_wait_ms,
        "crawlTemplateCap": primer.capture.template_page_cap,
        "crawlOpenMenus": primer.capture.open_menus,
        "crawlAdditionalPaths": primer.capture.additional_paths,
        "palettes": [],
        "connectors": connectors,
        "referenceImages": [],
    }


def build_project_primer_package(
    primer: ProjectPrimer,
    *,
    preview: AgentPreview | None = None,
    preview_root: Path | None = None,
) -> bytes:
    """Create a secret-free `.monetproj`, optionally with a rendered preview."""

    from io import BytesIO

    now = datetime.now(UTC)
    preview_assets: list[tuple[AgentPreviewPage, Path, int, int]] = []
    if preview is not None:
        if preview_root is None:
            raise PrimerError("Agent Preview packaging requires its manifest folder.")
        preview_assets = _validated_preview_assets(primer, preview, preview_root)

    manifest = {
        "formatVersion": FORMAT_VERSION,
        "exportedAt": now.isoformat(),
        "exportedByID": "monet-project-primer",
        "exportedByName": primer.generated_by.name,
        "appBuild": None,
        "versionCount": 1 if preview is not None else 0,
    }
    out = BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr(
            "project.json",
            json.dumps(_project_dto(primer), ensure_ascii=False, indent=2),
        )
        archive.writestr("primer.json", primer.canonical_json())
        if preview is not None:
            pages = []
            for review_order, (page, screenshot, image_width, image_height) in enumerate(
                preview_assets
            ):
                pages.append(
                    {
                        "url": page.url,
                        "pageSlug": page.page_slug,
                        "reviewOrder": review_order,
                        "pageTitle": page.page_title,
                        "viewportW": page.viewport_width or image_width,
                        "viewportH": page.viewport_height,
                        "scrollHeight": page.scroll_height or image_height,
                        "capturedAt": preview.captured_at.isoformat(),
                        "metaDescription": page.meta_description,
                        "ogTitle": None,
                        "ogDescription": None,
                        "ogImage": None,
                        "canonicalURL": page.canonical_url,
                        "h1": page.h1,
                        "seoNotes": None,
                        "hasDOMMetadata": False,
                        "hasDOMElementMap": None,
                        "hasFreehand": False,
                        "notes": [],
                        "annotations": [],
                    }
                )
                archive.write(
                    screenshot,
                    f"versions/hermes-preview/pages/{page.page_slug}.png",
                )

            provenance = (
                "Rendered by Hermes for immediate review. Refresh from the website in "
                "Monet before using this version as a verification baseline."
            )
            if preview.notes:
                provenance = f"{provenance}\n\n{preview.notes}"
            version = {
                "label": preview.label,
                "capturedAt": preview.captured_at.isoformat(),
                "notes": provenance,
                "viewport": preview.viewport.value,
                "colorScheme": preview.color_scheme.value,
                "capturedWith": preview.captured_with.value,
                "captureKind": AGENT_PREVIEW_CAPTURE_KIND,
                "parentVersionLabel": None,
                "pages": pages,
            }
            archive.writestr(
                "versions/hermes-preview/version.json",
                json.dumps(version, ensure_ascii=False, indent=2),
            )
    return out.getvalue()


def write_project_primer_package(
    primer: ProjectPrimer,
    output: Path,
    *,
    preview: AgentPreview | None = None,
    preview_root: Path | None = None,
) -> Path:
    """Write one validated Primer package and return its final path."""

    if output.suffix.lower() != ".monetproj":
        output = output / f"{primer.project.slug}.monetproj"
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temp.write_bytes(
            build_project_primer_package(
                primer,
                preview=preview,
                preview_root=preview_root,
            )
        )
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    return output.resolve()


def encode_setup_url(primer: ProjectPrimer) -> str | None:
    """Encode a small secret-free Primer in a URL fragment for direct opening.

    Fragments are not sent to iammonet.com. Oversized configurations return
    ``None`` and should travel as a `.monetproj` file.
    """

    compressed = zlib.compress(primer.canonical_json(), level=9)
    payload = base64.urlsafe_b64encode(compressed).decode("ascii").rstrip("=")
    url = f"{PRIMER_SETUP_URL}#p1.{payload}"
    return url if len(url) <= MAX_INLINE_SETUP_URL_CHARS else None


def decode_setup_url(url: str) -> ProjectPrimer:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "iammonet.com" or parsed.path != "/setup":
        raise PrimerError("That is not a Monet setup link.")
    if not parsed.fragment.startswith("p1."):
        raise PrimerError("Unsupported Monet setup link version.")
    raw = parsed.fragment[3:]
    try:
        padded = raw + "=" * (-len(raw) % 4)
        compressed = base64.urlsafe_b64decode(padded)
        decoder = zlib.decompressobj()
        body = decoder.decompress(compressed, MAX_PRIMER_JSON_BYTES + 1)
        body += decoder.flush()
    except (ValueError, zlib.error) as error:
        raise PrimerError("Monet setup link is damaged.") from error
    if len(body) > MAX_PRIMER_JSON_BYTES or decoder.unconsumed_tail:
        raise PrimerError("Monet setup link is too large.")
    return load_project_primer(body)


def desktop_installation_status(system: str | None = None) -> dict[str, Any]:
    """Detect supported desktop installs without launching or modifying them."""

    system = system or platform.system()
    if system == "Darwin":
        candidates = [Path("/Applications/Monet.app"), Path.home() / "Applications/Monet.app"]
        installed = next((path for path in candidates if path.exists()), None)
        return {
            "platform": "macos",
            "supported": True,
            "installed": bool(installed),
            "path": str(installed) if installed else None,
            "downloadURL": "https://iammonet.com/buy#desktop-download",
        }
    if system == "Windows":
        roots: list[Path] = []
        if local_app_data := os.environ.get("LOCALAPPDATA"):
            roots.extend(
                [
                    Path(local_app_data) / "Programs/Monet/Monet.exe",
                    Path(local_app_data) / "Monet/Monet.exe",
                ]
            )
        if program_files := os.environ.get("PROGRAMFILES"):
            roots.append(Path(program_files) / "Monet/Monet.exe")
        installed = next((path for path in roots if str(path) and path.exists()), None)
        return {
            "platform": "windows",
            "supported": True,
            "installed": bool(installed),
            "path": str(installed) if installed else None,
            "downloadURL": "https://iammonet.com/buy#desktop-download",
        }
    return {
        "platform": system.lower() or "unknown",
        "supported": False,
        "installed": False,
        "path": None,
        "downloadURL": None,
        "reason": "Monet Desktop supports macOS and Windows. Use Monet for iPad otherwise.",
    }


def open_package_in_monet(package: Path) -> None:
    """Open a package with the OS-associated Monet app after user consent."""

    status = desktop_installation_status()
    if not status["supported"]:
        raise PrimerError(status["reason"])
    if not status["installed"]:
        raise PrimerError("Monet Desktop is not installed.")
    if status["platform"] == "macos":
        subprocess.run(["open", str(package)], check=True)  # noqa: S603,S607
    elif status["platform"] == "windows":
        os.startfile(str(package))  # type: ignore[attr-defined]  # noqa: S606


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Build a secret-free Monet Project Primer.")
    parser.add_argument("input", type=Path, help="Project Primer JSON")
    parser.add_argument("--output", type=Path, default=Path.cwd())
    parser.add_argument("--preview", type=Path, help="Optional Agent Preview JSON")
    parser.add_argument("--open", action="store_true", dest="open_package")
    args = parser.parse_args()
    try:
        primer = load_project_primer(args.input)
        preview = load_agent_preview(args.preview) if args.preview else None
        package = write_project_primer_package(
            primer,
            args.output,
            preview=preview,
            preview_root=args.preview.parent if args.preview else None,
        )
        result = {
            "package": str(package),
            "setupURL": encode_setup_url(primer),
            "desktop": desktop_installation_status(),
            "recommendedSurface": "ipad",
            "previewPages": len(preview.pages) if preview else 0,
        }
        print(json.dumps(result, indent=2))
        if args.open_package:
            open_package_in_monet(package)
        return 0
    except PrimerError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
