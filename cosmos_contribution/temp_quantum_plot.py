import json
import matplotlib.pyplot as plt
import os

ARCHIVAL_PATH = r"D:\Cosmos\data\archival\quantum_runs.jsonl"
OUTPUT_DIR = r"C:\Users\corys\.gemini\antigravity\brain\e39fa9c0-1407-4d84-9683-4edbfa79866b"

def generate_graphs():
    if not os.path.exists(ARCHIVAL_PATH):
        print("No quantum run data found.")
        return

    runs = []
    with open(ARCHIVAL_PATH, 'r') as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))
                
    print(f"Decoded {len(runs)} quantum jobs.")
    
    # 1. Aggregate Histogram of Bitstrings
    all_counts = {}
    for run in runs:
        for bitstring, count in run.get("counts", {}).items():
            all_counts[bitstring] = all_counts.get(bitstring, 0) + count
            
    # Sort and plot top 20 bitstrings
    sorted_counts = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    states = [k for k, v in sorted_counts]
    counts = [v for k, v in sorted_counts]
    
    plt.figure(figsize=(12, 6))
    plt.bar(states, counts, color='cyan', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.title("Aggregated Quantum Entanglement States Across All Runs", color='white')
    plt.xlabel("Measured Quantum State (Bitstring)", color='white')
    plt.ylabel("Frequency", color='white')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Customize for dark theme
    ax = plt.gca()
    ax.set_facecolor('#1e1e1e')
    plt.gcf().set_facecolor('#1e1e1e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    out_path = os.path.join(OUTPUT_DIR, "aggregated_quantum_histogram.png")
    plt.tight_layout()
    plt.savefig(out_path, facecolor='#1e1e1e')
    print(f"Saved aggregated histogram to {out_path}")
    
    # Generate an individual plot for each run's physics metrics (Resonance, Emotion, Complexity)
    times = [i for i in range(len(runs))]
    resonances = [r.get("physics", {}).get("resonance", 0) for r in runs]
    
    res_path = os.path.join(OUTPUT_DIR, "quantum_resonance_trend.png")
    if times and resonances:
        plt.figure(figsize=(10, 5))
        plt.plot(times, resonances, marker='o', linestyle='-', color='magenta')
        plt.title("User Resonance Stability Over Time", color='white')
        plt.xlabel("Quantum Run Index", color='white')
        plt.ylabel("12D Resonance Factor", color='white')
        ax = plt.gca()
        ax.set_facecolor('#1e1e1e')
        plt.gcf().set_facecolor('#1e1e1e')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            
        plt.savefig(res_path, facecolor='#1e1e1e')
        print(f"Saved resonance trend to {res_path}")

if __name__ == "__main__":
    generate_graphs()
