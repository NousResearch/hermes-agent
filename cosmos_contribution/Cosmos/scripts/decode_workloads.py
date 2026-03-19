import json
import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from qiskit_ibm_runtime import RuntimeDecoder

def decode_workloads(folder_path, output_image):
    print(f"Loading workloads from {folder_path}...")
    
    result_files = glob.glob(os.path.join(folder_path, "*-result.json"))
    
    print(f"Found {len(result_files)} result files.")
    
    global_counts = defaultdict(int)
    total_shots = 0
    total_jobs = len(result_files)
    
    for res_file in result_files:
        try:
            with open(res_file, "r") as f:
                # Use Qiskit's native decoder
                primitive_result = json.load(f, cls=RuntimeDecoder)
                
            # V2 Primitives format: result[0].data.meas.get_counts()
            pub_result = primitive_result[0]
            counts = pub_result.data.meas.get_counts()
            
            for state, count in counts.items():
                global_counts[state] += count
            total_shots += sum(counts.values())
        except Exception as e:
            print(f"Failed to parse {res_file}: {e}")

    if total_shots == 0:
        print("No valid quantum shot data found to plot.")
        return

    print(f"Successfully decoded {total_jobs} jobs with {total_shots} total shots.")
    
    # Analyze state distributions across all runs
    max_len = 5 # Target 5 qubits
    all_states = [format(i, f'0{max_len}b') for i in range(2**max_len)]
    frequencies = {state: global_counts.get(state, 0) for state in all_states}
    
    # Sort for display
    sorted_states = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    states = [s[0] for s in sorted_states]
    counts = [s[1] for s in sorted_states]
    probabilities = [(count / total_shots) * 100 for count in counts]
    
    print("\nTop 10 Quantum Entanglement States in Workloads:")
    for state, count in sorted_states[:10]:
        print(f"|{state}⟩ : {count} ({count/total_shots*100:.2f}%)")

    # Plotting
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = plt.cm.plasma(np.linspace(0, 1, len(states)))
    bars = ax.bar(states, probabilities, color=colors, edgecolor='magenta', alpha=0.8)

    ax.set_title(f"IBM Quantum Workloads Decode ({total_jobs} Jobs | {total_shots:,} Shots)", 
                 fontsize=18, pad=20, color='white', fontweight='bold')
    ax.set_xlabel(f"Hilbert Space Basis States ({max_len} Qubits)", fontsize=14, labelpad=15, color='white')
    ax.set_ylabel("Measurement Probability (%)", fontsize=14, labelpad=15, color='white')
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#aaaaaa')
    ax.spines['bottom'].set_color('#aaaaaa')

    plt.xticks(rotation=90, fontsize=10, fontfamily='monospace')
    plt.yticks(fontsize=12)
    
    # Add value labels on top of the top 5 bars
    for i in range(5):
        bar = bars[i]
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color='yellow', fontweight='bold')

    # Add watermark/footer
    plt.figtext(0.99, 0.01, 'Decoded by Cosmos 12D Hebbian Transformer | Hardware: Real IBM Cloud',
                horizontalalignment='right', fontsize=9, color='#888888')

    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\nWorkload graph successfully saved to {output_image}")

if __name__ == "__main__":
    folder = r"D:\Cosmos\Cosmos\workloads\workloads (4)"
    out_img = r"D:\Cosmos\Cosmos\workload_histogram.png"
    decode_workloads(folder, out_img)
