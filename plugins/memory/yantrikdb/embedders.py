"""Built-in embedder wrappers for the embedded backend.

v0.4.2 adds two first-class loaders so users don't have to write a thin
wrapper class for the two most common embedder families:

  * :class:`Model2VecEmbedder` — wraps ``model2vec.StaticModel`` for the
    lightweight static-embedding family (``minishlab/potion-base-*M``,
    ``minishlab/potion-multilingual-128M``, etc.). Fast, no torch dep.
  * :class:`SentenceTransformerEmbedder` — wraps
    ``sentence_transformers.SentenceTransformer`` for the broader HF
    ecosystem (``all-MiniLM-L6-v2``, ``paraphrase-multilingual-MiniLM-L12-v2``,
    ``all-mpnet-base-v2``, etc.).

Both pick up their heavy dependency lazily (imported inside ``__init__``)
so this module can be imported without either package installed — the
loader only fails when a user actually configures that path. Both also
probe the output dim once on construction (calling
``.encode("__yantrikdb_probe__")`` and using ``len()`` of the result) so
the plugin can pass the right ``embedding_dim`` to ``YantrikDB(...)``
without the user having to set ``YANTRIKDB_EMBEDDING_DIM`` for the
builtin paths.

Both wrappers expose ``.encode(text: str) -> list[float]`` — the engine's
:py:meth:`set_embedder` contract — and have a public ``.embedding_dim``
attribute so :mod:`.embedded` can read the probed dim back out.
"""

from __future__ import annotations

import logging
from typing import Any

from .client import YantrikDBError

logger = logging.getLogger(__name__)

_PROBE_TEXT = "__yantrikdb_probe__"


def _coerce_to_float_list(vec: Any, *, model_name: str, family: str) -> list[float]:
    """Normalise the embedder's encode() return into ``list[float]``.

    Different libraries return numpy arrays, torch tensors, or plain
    lists. The engine's ``set_embedder`` expects a Python ``list[float]``
    (per the pyo3 binding). We accept any 1-D sequence-like and convert.
    """
    # numpy ndarray / torch tensor / list / tuple — all support tolist() or iter
    out = vec.tolist() if hasattr(vec, "tolist") else list(vec)
    # Some libs return a 2-D batch even when called with a single string;
    # flatten one level if needed.
    if out and isinstance(out[0], list):
        out = out[0]
    try:
        return [float(x) for x in out]
    except (TypeError, ValueError) as e:
        raise YantrikDBError(
            f"{family} loader for {model_name!r} returned a non-numeric vector "
            f"from .encode(): {type(vec).__name__}. Cannot use as embedder."
        ) from e


class Model2VecEmbedder:
    """Wraps ``model2vec.StaticModel`` for use with ``YantrikDB.set_embedder``.

    Constructs the underlying ``StaticModel.from_pretrained(model_name)``
    once. Subsequent ``.encode(text)`` calls reuse the loaded model.

    The output dim is probed once at construction time and exposed as
    :attr:`embedding_dim` so the plugin can pass it to ``YantrikDB(...)``.
    """

    def __init__(self, model_name: str) -> None:
        try:
            from model2vec import StaticModel  # type: ignore[import-untyped]
        except ImportError as e:
            raise YantrikDBError(
                "YANTRIKDB_EMBEDDER_MODEL2VEC requires the `model2vec` package. "
                "Install with: pip install 'yantrikdb-hermes-plugin[model2vec]' "
                "  (or: pip install model2vec)"
            ) from e
        self.model_name = model_name
        try:
            self._model = StaticModel.from_pretrained(model_name)
        except Exception as e:
            raise YantrikDBError(
                f"model2vec failed to load model {model_name!r}: {e}. "
                "Check the model name on Hugging Face and that you have "
                "network access on first run (subsequent runs use the HF cache)."
            ) from e
        # Probe dim once — guarantees the loader is healthy before we
        # hand it to the engine and lets us infer the right embedding_dim.
        probe = _coerce_to_float_list(
            self._model.encode([_PROBE_TEXT])[0],
            model_name=model_name, family="model2vec",
        )
        self.embedding_dim = len(probe)
        if self.embedding_dim <= 0:
            raise YantrikDBError(
                f"model2vec {model_name!r} probed to an empty vector. "
                "This is almost certainly a broken model load."
            )
        logger.info(
            "Model2VecEmbedder loaded: %s (dim=%d)",
            model_name, self.embedding_dim,
        )

    def encode(self, text: str) -> list[float]:
        # StaticModel.encode accepts a single string or a list; calling
        # with a list always returns a 2-D ndarray, calling with a string
        # is version-dependent. The list form is the stable contract.
        out = self._model.encode([text])
        # out is shape (1, dim); take the first row.
        row = out[0] if hasattr(out, "__len__") and len(out) else out
        return _coerce_to_float_list(row, model_name=self.model_name, family="model2vec")


class SentenceTransformerEmbedder:
    """Wraps ``sentence_transformers.SentenceTransformer`` for use with
    ``YantrikDB.set_embedder``.

    Pulls in PyTorch transitively. Heavier than :class:`Model2VecEmbedder`
    but covers the broader HF embedder ecosystem. Output dim is probed
    once at construction.
    """

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as e:
            raise YantrikDBError(
                "YANTRIKDB_EMBEDDER_HF requires the `sentence-transformers` "
                "package. Install with: "
                "pip install 'yantrikdb-hermes-plugin[sentence-transformers]' "
                "  (or: pip install sentence-transformers)"
            ) from e
        self.model_name = model_name
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as e:
            raise YantrikDBError(
                f"sentence-transformers failed to load model {model_name!r}: {e}. "
                "Check the model name on Hugging Face and that you have "
                "network access on first run (subsequent runs use the HF cache)."
            ) from e
        # Prefer the model's declared dim if available — saves an encode
        # call and matches what HF says rather than relying on probe.
        declared = getattr(self._model, "get_sentence_embedding_dimension", None)
        if callable(declared):
            try:
                self.embedding_dim = int(declared())
            except Exception:
                self.embedding_dim = 0
        else:
            self.embedding_dim = 0
        # Always probe — confirms the loader actually works before we
        # hand it to the engine and as a fallback when declared dim isn't
        # available.
        probe = _coerce_to_float_list(
            self._model.encode(_PROBE_TEXT),
            model_name=model_name, family="sentence-transformers",
        )
        probed = len(probe)
        if self.embedding_dim and self.embedding_dim != probed:
            logger.warning(
                "sentence-transformers %s: declared dim=%d but probe returned %d; "
                "using probed value.",
                model_name, self.embedding_dim, probed,
            )
        self.embedding_dim = probed
        if self.embedding_dim <= 0:
            raise YantrikDBError(
                f"sentence-transformers {model_name!r} probed to an empty vector."
            )
        logger.info(
            "SentenceTransformerEmbedder loaded: %s (dim=%d)",
            model_name, self.embedding_dim,
        )

    def encode(self, text: str) -> list[float]:
        # SentenceTransformer.encode with a single string returns a 1-D
        # ndarray of shape (dim,). With a list it returns 2-D.
        vec = self._model.encode(text)
        return _coerce_to_float_list(
            vec, model_name=self.model_name, family="sentence-transformers",
        )


__all__ = ["Model2VecEmbedder", "SentenceTransformerEmbedder"]
