import json
import matplotlib.pyplot as plt
from pathlib import Path
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime.utils.json import RuntimeDecoder

def aggregate_jobs():
    workload_dir = Path(r"D:\Cosmos\Cosmos\workloads\workloads (2)")
    info_files = list(workload_dir.glob("*-info.json"))
    
    total_counts = {}
    job_entropies = []
    
    for info_file in info_files:
        res_file = workload_dir / info_file.name.replace("-info", "-result")
        if not res_file.exists():
            continue
            
        with open(res_file, 'r') as f:
            res = json.load(f, cls=RuntimeDecoder)
            
        try:
            counts = res[0].data.meas.get_counts()
            
            # Aggregate counts over all jobs
            for bitstring, count in counts.items():
                total_counts[bitstring] = total_counts.get(bitstring, 0) + count
        except Exception as e:
            print(f"Skipping {info_file.name} due to structure differences: {e}")

    if not total_counts:
        print("No valid counts found across any jobs.")
        return

    # Sort aggregated counts
    sorted_total = dict(sorted(total_counts.items(), key=lambda item: item[1], reverse=True)[:25])

    # Plot massive aggregated histogram
    hist_path = r"D:\Cosmos\Cosmos\aggregated_quantum_histogram.png"
    fig = plot_histogram(sorted_total, title="Aggregated Superposition Density (19 Quantum Jobs, 77,824 Shots)", color='#7d3c98', figsize=(14,7))
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    print(f"SUCCESS: Saved aggregated graph to {hist_path}")

if __name__ == "__main__":
    aggregate_jobs()
