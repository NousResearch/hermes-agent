from hermes_cli.local_runtime import (
    discover_ollama_manifest_models,
    parse_lmstudio_ls_output,
    parse_ollama_list_output,
)


def test_parse_lmstudio_ls_filters_embeddings_by_default():
    output = """
You have 3 models, taking up 25.04 GB of disk space.

LLM                                 PARAMS     ARCH        SIZE        DEVICE
qwen/qwen3-coder-30b (1 variant)    30B-A3B    qwen3moe    18.63 GB    Local

EMBEDDING                               PARAMS    ARCH          SIZE        DEVICE
text-embedding-nomic-embed-text-v1.5              Nomic BERT    84.11 MB    Local
"""

    models = parse_lmstudio_ls_output(output)

    assert [m.id for m in models] == ["qwen/qwen3-coder-30b"]
    assert models[0].backend == "lmstudio"
    assert models[0].params == "30B-A3B"
    assert models[0].architecture == "qwen3moe"
    assert models[0].size == "18.63 GB"


def test_parse_lmstudio_ls_can_include_embeddings():
    output = """
LLM                                 PARAMS     ARCH        SIZE        DEVICE
google/gemma-4-e4b (1 variant)      7.5B       gemma4      6.33 GB     Local

EMBEDDING                               PARAMS    ARCH          SIZE        DEVICE
text-embedding-nomic-embed-text-v1.5              Nomic BERT    84.11 MB    Local
"""

    models = parse_lmstudio_ls_output(output, include_embeddings=True)

    assert [m.id for m in models] == [
        "google/gemma-4-e4b",
        "text-embedding-nomic-embed-text-v1.5",
    ]
    assert models[1].kind == "embedding"


def test_parse_ollama_list_output():
    output = """
NAME              ID              SIZE      MODIFIED
qwen3:14b         abcdef123456    9.3 GB    2 days ago
llama3.2:latest   123456abcdef    2.0 GB    1 week ago
"""

    models = parse_ollama_list_output(output)

    assert [m.id for m in models] == ["qwen3:14b", "llama3.2:latest"]
    assert models[0].digest == "abcdef123456"
    assert models[0].size == "9.3 GB"


def test_discover_ollama_manifest_models_without_daemon(tmp_path):
    manifests = tmp_path / "models" / "manifests"
    manifest = manifests / "registry.ollama.ai" / "library" / "qwen3" / "14b"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"schemaVersion": 2}', encoding="utf-8")

    namespaced = manifests / "registry.ollama.ai" / "someuser" / "custom-model" / "latest"
    namespaced.parent.mkdir(parents=True)
    namespaced.write_text('{"schemaVersion": 2}', encoding="utf-8")

    models = discover_ollama_manifest_models(manifests)

    assert [m.id for m in models] == ["qwen3:14b", "someuser/custom-model:latest"]
    assert all(m.source == "ollama manifest" for m in models)
