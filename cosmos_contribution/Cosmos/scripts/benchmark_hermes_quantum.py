import json
import os
import time
from collections import defaultdict
import numpy as np

def load_archival_runs(filepath="data/archival/quantum_runs.jsonl"):
    runs = []
    if not os.path.exists(filepath):
        print(f"Archival file {filepath} not found.")
        return runs
        
    with open(filepath, "r") as f:
        for line in f:
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return runs

def compute_entropy_quality(counts):
        if not counts: return 0.0
        
        total = sum(counts.values())
        if total == 0: return 0.0
        
        probs = [count / total for count in counts.values() if count > 0]
        # Max entropy for 5 qubits is ln(32) ≈ 3.465
        entropy = -sum(p * np.log(p) for p in probs)
        max_entropy = np.log(len(counts)) if len(counts) > 1 else 1.0
        
        return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0

def analyze_runs(runs):
    output_lines = []
    
    if not runs:
        output_lines.append("No runs to analyze.")
        with open("quantum_benchmark_results.txt", "w") as f:
            f.write("\n".join(output_lines))
        return

    output_lines.append(f"\n=================================================")
    output_lines.append(f" 🧠 HERMES QUANTUM SYMBIOSIS BENCHMARK ({len(runs)} RUNS) ")
    output_lines.append(f"=================================================\n")

    # Overall stats
    total_shots = sum(run.get("total_shots", 0) for run in runs)
    avg_shots = total_shots / len(runs)
    
    # Analyze state distributions across all runs
    global_counts = defaultdict(int)
    for run in runs:
        counts = run.get("counts", {})
        for state, count in counts.items():
            global_counts[state] += count
            
    # Calculate Quality metrics
    qualities = [compute_entropy_quality(r.get("counts", {})) for r in runs]
    avg_quality = sum(qualities) / len(qualities)
            
    # Top 10 states
    sorted_states = sorted(global_counts.items(), key=lambda x: x[1], reverse=True)
    top_10 = sorted_states[:10]

    # Create summary table
    output_lines.append("Top 10 Quantum Entanglement States Across All Runs")
    output_lines.append(f"{'State':<15} | {'Frequency':<15} | {'Probability':<15}")
    output_lines.append("-" * 50)
    
    for state, count in top_10:
        pct = (count / total_shots) * 100 if total_shots > 0 else 0
        output_lines.append(f"|{state}⟩           | {count:<15} | {pct:.2f}%")
        
    output_lines.append("\nOverall Metrics:")
    output_lines.append(f"Total Shots Processed: {total_shots:,}")
    output_lines.append(f"Average Shots per Run: {avg_shots:,.1f}")
    output_lines.append(f"Unique Basis States Reached: {len(global_counts)}/32")
    output_lines.append(f"Average Entropy Quality Signal feed to RL: {avg_quality:.4f}")
    
    # Physics ranges
    if any("physics" in r for r in runs):
        output_lines.append("\nEmotional Physics Input Dynamics:")
        phases = [r.get("physics", {}).get("cst_physics", {}).get("geometric_phase_rad", 0.0) for r in runs]
        entropies = [r.get("physics", {}).get("virtual_body", {}).get("entropy", 0.5) for r in runs]
        resonance = [r.get("physics", {}).get("virtual_body", {}).get("resonance", 0.0) for r in runs]
        
        # fallback if cst_physics didn't catch
        if all(p == 0.0 for p in phases):
            phases = [r.get("physics", {}).get("geometric_phase_rad", 0.0) for r in runs]
            entropies = [r.get("physics", {}).get("entropy_field", 0.5) for r in runs]
            resonance = [r.get("physics", {}).get("resonance_scalar", 0.0) for r in runs]
        
        output_lines.append(f"CST Phase Modulation Range: [min: {np.min(phases):.3f}, max: {np.max(phases):.3f}, avg: {np.mean(phases):.3f}] rad")
        output_lines.append(f"Swarm Entropy Range:        [min: {np.min(entropies):.3f}, max: {np.max(entropies):.3f}, avg: {np.mean(entropies):.3f}]")
        output_lines.append(f"Resonance Scale Range:      [min: {np.min(resonance):.3f}, max: {np.max(resonance):.3f}, avg: {np.mean(resonance):.3f}]")

    result_text = "\n".join(output_lines)
    print("Benchmark complete. Written to quantum_benchmark_results.txt")
    
    with open("quantum_benchmark_results.txt", "w", encoding="utf-8") as f:
        f.write(result_text)

if __name__ == "__main__":
    runs = load_archival_runs(r"d:\Cosmos\data\archival\quantum_runs.jsonl")
    analyze_runs(runs)
