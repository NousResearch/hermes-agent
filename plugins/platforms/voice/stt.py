"""Streaming STT port + adapter factory (Ports & Adapters).

The voice turn loop consumes a provider-agnostic stream of normalized turn
events; this module defines that contract and the factory that builds a
concrete adapter for it.

The canonical home of the normalized event names is HERE — every adapter
(``deepgram_flux_stt``, ``cartesia_ink_stt``) maps its provider-specific wire
events onto these constants, and the turn loop branches only on these. They are
re-exported by ``deepgram_flux_stt`` so existing imports keep working.

  start_of_turn      -> the user began speaking (barge-in trigger)
  update             -> incremental partial transcript
  eager_end_of_turn  -> medium-confidence end -> start the agent speculatively
  turn_resumed       -> the user kept talking past the eager signal -> cancel
  end_of_turn        -> high-confidence end -> commit the turn

Default provider is Deepgram Flux. Cartesia Ink-2 is a second adapter behind
the same port for A/B comparison (turn-taking + latency); accuracy is out of
scope for the seam.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Protocol, runtime_checkable

# Normalized event names the turn loop consumes (provider-agnostic). Canonical
# home — adapters import these from here.
EV_START = "start_of_turn"
EV_UPDATE = "update"
EV_EAGER_EOT = "eager_end_of_turn"
EV_RESUMED = "turn_resumed"
EV_END = "end_of_turn"


@runtime_checkable
class STT(Protocol):
    """The streaming STT port. An adapter holds one provider websocket for a
    call: ``send_audio`` streams mic PCM in, ``events()`` yields normalized turn
    events out. ``provider`` tags telemetry for the A/B.

    Normalized event shape (every yield):
        {"event": <EV_*>, "transcript": str, "confidence": float | None,
         "words": list, "turn_index": int | None}
    """

    #: Stable provider id used to tag telemetry (e.g. "deepgram_flux").
    provider: str

    @property
    def asr_seconds_est(self) -> float:
        """Estimated seconds of audio streamed to ASR (per-call cost record)."""
        ...

    async def start(self) -> None:
        """Open the provider websocket and begin receiving."""
        ...

    async def send_audio(self, pcm: bytes) -> None:
        """Stream one chunk of s16le mono PCM at the declared input rate."""
        ...

    def events(self) -> AsyncIterator[Dict[str, Any]]:
        """Yield normalized turn events until the stream ends."""
        ...

    async def configure(self, **kw: Any) -> None:
        """Update provider turn-detection settings mid-stream, if supported."""
        ...

    async def stop(self) -> None:
        """Close the provider websocket and tear down."""
        ...


DEFAULT_STT_PROVIDER = "deepgram_flux"


def make_stt(provider: str, *, input_rate: int, extra: Dict[str, Any]) -> STT:
    """Build the STT adapter for ``provider``.

    Resolves the provider's API key from the environment (``DEEPGRAM_API_KEY`` /
    ``CARTESIA_API_KEY``), passes Flux its turn-detection thresholds from
    ``extra``, and raises a clear error for an unknown provider. The adapter
    constructors raise on a missing key, so a misconfigured provider fails fast.

    Sibling modules are imported the discord dual-import way (flat for the test
    loader, relative for the production ``hermes_plugins.platforms__voice``
    package) so this works in both contexts.
    """
    import os

    provider = (provider or DEFAULT_STT_PROVIDER).strip().lower()

    if provider == "deepgram_flux":
        try:
            import deepgram_flux_stt
        except ImportError:
            from . import deepgram_flux_stt
        dg_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
        if not dg_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")
        return deepgram_flux_stt.DeepgramFluxSTT(
            dg_key,
            input_rate=input_rate,
            eot_threshold=_float_extra(
                extra, "eot_threshold",
                deepgram_flux_stt.DEFAULT_EOT_THRESHOLD),
            eager_eot_threshold=_float_extra(
                extra, "eager_eot_threshold",
                deepgram_flux_stt.DEFAULT_EAGER_EOT_THRESHOLD),
        )

    if provider == "cartesia_ink":
        try:
            import cartesia_ink_stt
        except ImportError:
            from . import cartesia_ink_stt
        ck_key = os.getenv("CARTESIA_API_KEY", "").strip()
        if not ck_key:
            raise RuntimeError("CARTESIA_API_KEY is not set")
        return cartesia_ink_stt.CartesiaInkSTT(ck_key, input_rate=input_rate)

    raise RuntimeError(
        f"unknown stt_provider={provider!r}; "
        "supported: 'deepgram_flux' (default), 'cartesia_ink'")


def _float_extra(extra: Dict[str, Any], key: str, default: float) -> float:
    """Parse a float from extra, falling back to the default on missing/bad
    input (no silent surprises — matches adapter._resolve_float_extra)."""
    raw = (extra or {}).get(key)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default
