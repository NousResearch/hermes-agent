#!/usr/bin/env python3
"""
Farnsworth Benchmark Suite

Benchmarks system performance:
- Memory operations (store, recall, search)
- Agent task execution
- Embedding generation
- LLM inference speed
- End-to-end response time
"""

import argparse
import asyncio
import json
import time
import statistics
from datetime import datetime
from pathlib import Path
from typing import Optional
import tempfile


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str):
        self.name = name
        self.times: list[float] = []
        self.successes: int = 0
        self.failures: int = 0
        self.metadata: dict = {}

    def add_sample(self, duration: float, success: bool = True):
        """Add a sample to the benchmark."""
        self.times.append(duration)
        if success:
            self.successes += 1
        else:
            self.failures += 1

    @property
    def mean(self) -> float:
        return statistics.mean(self.times) if self.times else 0

    @property
    def median(self) -> float:
        return statistics.median(self.times) if self.times else 0

    @property
    def std_dev(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0

    @property
    def min_time(self) -> float:
        return min(self.times) if self.times else 0

    @property
    def max_time(self) -> float:
        return max(self.times) if self.times else 0

    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "samples": len(self.times),
            "mean_ms": self.mean * 1000,
            "median_ms": self.median * 1000,
            "std_dev_ms": self.std_dev * 1000,
            "min_ms": self.min_time * 1000,
            "max_ms": self.max_time * 1000,
            "success_rate": self.success_rate,
            "metadata": self.metadata,
        }


def print_header(text: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_result(result: BenchmarkResult):
    """Print benchmark result."""
    print(f"\n  {result.name}")
    print(f"  {'─'*40}")
    print(f"  Samples:     {len(result.times)}")
    print(f"  Mean:        {result.mean*1000:.2f} ms")
    print(f"  Median:      {result.median*1000:.2f} ms")
    print(f"  Std Dev:     {result.std_dev*1000:.2f} ms")
    print(f"  Min:         {result.min_time*1000:.2f} ms")
    print(f"  Max:         {result.max_time*1000:.2f} ms")
    print(f"  Success:     {result.success_rate*100:.1f}%")


async def benchmark_memory_store(data_dir: str, iterations: int = 100) -> BenchmarkResult:
    """Benchmark memory storage operations."""
    from farnsworth.memory.memory_system import MemorySystem

    result = BenchmarkResult("Memory Store")

    with tempfile.TemporaryDirectory() as tmpdir:
        memory = MemorySystem(data_dir=tmpdir)
        await memory.initialize()

        for i in range(iterations):
            content = f"Test memory content {i}. This is a sample piece of text for benchmarking."

            start = time.perf_counter()
            try:
                await memory.remember(content, importance=0.5)
                duration = time.perf_counter() - start
                result.add_sample(duration, success=True)
            except Exception as e:
                duration = time.perf_counter() - start
                result.add_sample(duration, success=False)

    return result


async def benchmark_memory_recall(data_dir: str, iterations: int = 50) -> BenchmarkResult:
    """Benchmark memory recall operations."""
    from farnsworth.memory.memory_system import MemorySystem

    result = BenchmarkResult("Memory Recall")

    with tempfile.TemporaryDirectory() as tmpdir:
        memory = MemorySystem(data_dir=tmpdir)
        await memory.initialize()

        # Seed some memories
        topics = ["Python programming", "Machine learning", "Data science",
                  "Web development", "System design", "Algorithms"]

        for i in range(100):
            topic = topics[i % len(topics)]
            await memory.remember(f"{topic} content {i}")

        # Benchmark recall
        queries = ["programming", "machine learning", "algorithms", "web", "data"]

        for i in range(iterations):
            query = queries[i % len(queries)]

            start = time.perf_counter()
            try:
                results = await memory.recall(query, top_k=5)
                duration = time.perf_counter() - start
                result.add_sample(duration, success=len(results) > 0)
            except Exception as e:
                duration = time.perf_counter() - start
                result.add_sample(duration, success=False)

    return result


async def benchmark_embedding_generation(iterations: int = 100) -> BenchmarkResult:
    """Benchmark embedding generation."""
    result = BenchmarkResult("Embedding Generation")

    try:
        from farnsworth.rag.embeddings import EmbeddingManager

        embedder = EmbeddingManager()

        texts = [
            "This is a short text.",
            "This is a medium length text that contains more words and information.",
            "This is a longer piece of text that simulates a more realistic document. " * 3,
        ]

        for i in range(iterations):
            text = texts[i % len(texts)]

            start = time.perf_counter()
            try:
                embedding = embedder.embed(text)
                duration = time.perf_counter() - start
                result.add_sample(duration, success=len(embedding) > 0)
            except Exception as e:
                duration = time.perf_counter() - start
                result.add_sample(duration, success=False)

        result.metadata["dimensions"] = len(embedding) if 'embedding' in dir() else 0

    except ImportError:
        result.metadata["error"] = "sentence-transformers not installed"

    return result


async def benchmark_llm_inference(iterations: int = 10) -> BenchmarkResult:
    """Benchmark LLM inference (if available)."""
    result = BenchmarkResult("LLM Inference")

    try:
        from farnsworth.core.llm_backend import OllamaBackend

        backend = OllamaBackend(model="deepseek-r1:1.5b")

        prompts = [
            "What is 2 + 2?",
            "Write a haiku about coding.",
            "Explain recursion in one sentence.",
        ]

        for i in range(iterations):
            prompt = prompts[i % len(prompts)]

            start = time.perf_counter()
            try:
                response = await backend.generate(prompt, max_tokens=50)
                duration = time.perf_counter() - start
                result.add_sample(duration, success=True)
                result.metadata["tokens_per_second"] = 50 / duration if duration > 0 else 0
            except Exception as e:
                duration = time.perf_counter() - start
                result.add_sample(duration, success=False)

    except Exception as e:
        result.metadata["error"] = str(e)

    return result


async def benchmark_knowledge_graph(iterations: int = 100) -> BenchmarkResult:
    """Benchmark knowledge graph operations."""
    from farnsworth.memory.knowledge_graph import KnowledgeGraph

    result = BenchmarkResult("Knowledge Graph")

    graph = KnowledgeGraph()

    # Add entities
    for i in range(iterations):
        start = time.perf_counter()
        try:
            entity_id = graph.add_entity(f"Entity_{i}", "TestType")
            if i > 0:
                graph.add_relationship(entity_id, f"entity_{i-1}", "related_to")
            duration = time.perf_counter() - start
            result.add_sample(duration, success=True)
        except Exception as e:
            duration = time.perf_counter() - start
            result.add_sample(duration, success=False)

    result.metadata["total_entities"] = len(graph.entities)
    result.metadata["total_relationships"] = len(graph.relationships)

    return result


async def benchmark_fitness_tracking(iterations: int = 1000) -> BenchmarkResult:
    """Benchmark fitness tracking operations."""
    from farnsworth.evolution.fitness_tracker import FitnessTracker

    result = BenchmarkResult("Fitness Tracking")

    tracker = FitnessTracker()

    for i in range(iterations):
        start = time.perf_counter()
        try:
            tracker.record("task_success", 0.8 + (i % 10) * 0.01)
            tracker.record("efficiency", 0.7 + (i % 10) * 0.02)
            duration = time.perf_counter() - start
            result.add_sample(duration, success=True)
        except Exception as e:
            duration = time.perf_counter() - start
            result.add_sample(duration, success=False)

    result.metadata["final_fitness"] = tracker.get_weighted_fitness()

    return result


async def run_all_benchmarks(data_dir: str, output_file: Optional[str] = None):
    """Run all benchmarks."""
    print_header("Farnsworth Benchmark Suite")
    print(f"\nStarted at: {datetime.now().isoformat()}")

    results = []

    # Memory benchmarks
    print("\n🧠 Memory System Benchmarks")

    result = await benchmark_memory_store(data_dir)
    results.append(result)
    print_result(result)

    result = await benchmark_memory_recall(data_dir)
    results.append(result)
    print_result(result)

    # Embedding benchmarks
    print("\n📊 Embedding Benchmarks")

    result = await benchmark_embedding_generation()
    results.append(result)
    print_result(result)

    # Knowledge graph benchmarks
    print("\n🕸️ Knowledge Graph Benchmarks")

    result = await benchmark_knowledge_graph()
    results.append(result)
    print_result(result)

    # Fitness tracking benchmarks
    print("\n📈 Evolution Benchmarks")

    result = await benchmark_fitness_tracking()
    results.append(result)
    print_result(result)

    # LLM benchmarks (optional)
    print("\n🤖 LLM Benchmarks")

    result = await benchmark_llm_inference()
    results.append(result)
    if result.times:
        print_result(result)
    else:
        print("  (Skipped - LLM not available)")

    # Summary
    print_header("Summary")

    total_samples = sum(len(r.times) for r in results)
    avg_success = statistics.mean([r.success_rate for r in results if r.times])

    print(f"\n  Total Benchmarks: {len(results)}")
    print(f"  Total Samples:    {total_samples}")
    print(f"  Avg Success Rate: {avg_success*100:.1f}%")

    # Save results
    if output_file:
        output = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": [r.to_dict() for r in results],
        }

        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n  Results saved to: {output_file}")

    print(f"\nCompleted at: {datetime.now().isoformat()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Farnsworth Benchmark Suite"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory for benchmarks"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks (fewer iterations)"
    )

    args = parser.parse_args()

    # Run benchmarks
    asyncio.run(run_all_benchmarks(
        data_dir=args.data_dir,
        output_file=args.output,
    ))


if __name__ == "__main__":
    main()
