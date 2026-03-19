#!/usr/bin/env python3
"""
Farnsworth Model Downloader

Downloads and configures local LLMs for Farnsworth:
- Ollama models
- GGUF models for llama.cpp
- Embedding models
"""

import argparse
import os
import sys
import subprocess
import urllib.request
import hashlib
from pathlib import Path
from typing import Optional

# Model configurations
OLLAMA_MODELS = {
    "deepseek-r1:1.5b": {
        "description": "DeepSeek-R1 Distilled 1.5B - Best reasoning at size",
        "size": "~1.2GB",
        "recommended": True,
    },
    "qwen3:0.6b": {
        "description": "Qwen3 0.6B - Ultra-lightweight",
        "size": "~400MB",
        "recommended": False,
    },
    "phi3:mini": {
        "description": "Phi-3 Mini - GPT-3.5 class reasoning",
        "size": "~2.4GB",
        "recommended": False,
    },
    "llama3.2:1b": {
        "description": "Llama 3.2 1B - Balanced performance",
        "size": "~700MB",
        "recommended": False,
    },
}

EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "description": "Fast, lightweight embeddings",
        "dimensions": 384,
        "recommended": True,
    },
    "all-mpnet-base-v2": {
        "description": "Higher quality embeddings",
        "dimensions": 768,
        "recommended": False,
    },
}


def print_banner():
    """Print download banner."""
    print("""
╔═══════════════════════════════════════════╗
║     Farnsworth Model Downloader           ║
╚═══════════════════════════════════════════╝
    """)


def check_ollama() -> bool:
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_ollama_model(model_name: str) -> bool:
    """Download a model using Ollama."""
    print(f"\n📦 Downloading {model_name}...")

    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            check=True,
        )
        print(f"✅ {model_name} downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False
    except FileNotFoundError:
        print("❌ Ollama not found. Install from: https://ollama.ai")
        return False


def download_embedding_model(model_name: str, data_dir: Path) -> bool:
    """Download embedding model using sentence-transformers."""
    print(f"\n📦 Downloading embedding model: {model_name}...")

    try:
        from sentence_transformers import SentenceTransformer

        # Download to cache
        model = SentenceTransformer(model_name)

        # Test it works
        test_embedding = model.encode("test")
        print(f"✅ {model_name} loaded (dimensions: {len(test_embedding)})")
        return True

    except ImportError:
        print("❌ sentence-transformers not installed")
        print("   Run: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False


def list_available_models():
    """List available models."""
    print("\n📋 Available LLM Models (via Ollama):\n")

    for name, info in OLLAMA_MODELS.items():
        rec = " ⭐ RECOMMENDED" if info.get("recommended") else ""
        print(f"  {name}{rec}")
        print(f"    {info['description']}")
        print(f"    Size: {info['size']}\n")

    print("\n📋 Available Embedding Models:\n")

    for name, info in EMBEDDING_MODELS.items():
        rec = " ⭐ RECOMMENDED" if info.get("recommended") else ""
        print(f"  {name}{rec}")
        print(f"    {info['description']}")
        print(f"    Dimensions: {info['dimensions']}\n")


def download_recommended(data_dir: Path) -> bool:
    """Download recommended models."""
    print("\n🚀 Downloading recommended models...\n")

    success = True

    # Check Ollama
    if check_ollama():
        print("✅ Ollama detected")

        # Download recommended Ollama model
        for name, info in OLLAMA_MODELS.items():
            if info.get("recommended"):
                if not download_ollama_model(name):
                    success = False
                break
    else:
        print("⚠️  Ollama not installed")
        print("   For best performance, install Ollama: https://ollama.ai")
        print("   Then run: ollama pull deepseek-r1:1.5b")

    # Download embedding model
    for name, info in EMBEDDING_MODELS.items():
        if info.get("recommended"):
            if not download_embedding_model(name, data_dir):
                success = False
            break

    return success


def interactive_setup(data_dir: Path):
    """Interactive model selection."""
    print("\n🎯 Interactive Model Setup\n")

    # LLM selection
    print("Select an LLM backend:\n")
    print("  1. Ollama (recommended - easy setup)")
    print("  2. llama.cpp (advanced - manual GGUF files)")
    print("  3. Skip LLM setup")

    choice = input("\nChoice [1]: ").strip() or "1"

    if choice == "1":
        if not check_ollama():
            print("\n⚠️  Ollama not installed.")
            print("Install from: https://ollama.ai")
            print("Then run this script again.")
        else:
            print("\nSelect a model:\n")
            models = list(OLLAMA_MODELS.keys())
            for i, name in enumerate(models, 1):
                info = OLLAMA_MODELS[name]
                rec = " ⭐" if info.get("recommended") else ""
                print(f"  {i}. {name}{rec} ({info['size']})")

            model_choice = input(f"\nChoice [1]: ").strip() or "1"
            try:
                model_idx = int(model_choice) - 1
                if 0 <= model_idx < len(models):
                    download_ollama_model(models[model_idx])
            except ValueError:
                print("Invalid choice")

    elif choice == "2":
        print("\n📝 llama.cpp Setup")
        print("1. Download a GGUF model from Hugging Face")
        print("2. Place it in:", data_dir / "models")
        print("3. Update configs/models.yaml with the path")

    # Embedding model
    print("\n\nSelect an embedding model:\n")
    models = list(EMBEDDING_MODELS.keys())
    for i, name in enumerate(models, 1):
        info = EMBEDDING_MODELS[name]
        rec = " ⭐" if info.get("recommended") else ""
        print(f"  {i}. {name}{rec}")

    emb_choice = input(f"\nChoice [1]: ").strip() or "1"
    try:
        emb_idx = int(emb_choice) - 1
        if 0 <= emb_idx < len(models):
            download_embedding_model(models[emb_idx], data_dir)
    except ValueError:
        print("Invalid choice")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download models for Farnsworth"
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--recommended", "-r",
        action="store_true",
        help="Download recommended models"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive model selection"
    )
    parser.add_argument(
        "--ollama",
        type=str,
        help="Download specific Ollama model"
    )
    parser.add_argument(
        "--embedding",
        type=str,
        help="Download specific embedding model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory"
    )

    args = parser.parse_args()

    print_banner()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_available_models()
    elif args.recommended:
        download_recommended(data_dir)
    elif args.interactive:
        interactive_setup(data_dir)
    elif args.ollama:
        download_ollama_model(args.ollama)
    elif args.embedding:
        download_embedding_model(args.embedding, data_dir)
    else:
        # Default: interactive setup
        interactive_setup(data_dir)

    print("\n✨ Model setup complete!")
    print("Run 'python main.py' to start Farnsworth")


if __name__ == "__main__":
    main()
