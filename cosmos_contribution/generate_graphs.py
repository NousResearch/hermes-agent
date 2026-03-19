import json
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit_ibm_runtime.utils.json import RuntimeDecoder

def generate_graphs():
    info_file = r"d:\Cosmos\workloads (2)\job-d6lmi8u9td6c73am3l60-info.json"
    res_file = r"d:\Cosmos\workloads (2)\job-d6lmi8u9td6c73am3l60-result.json"

    with open(info_file, 'r') as f:
        info = json.load(f, cls=RuntimeDecoder)
    circuit = info["params"]["pubs"][0][0]
    
    # Save the text drawing of the circuit
    with open(r"d:\Cosmos\circuit_drawing.txt", "w", encoding="utf-8") as text_file:
        text_file.write(str(circuit.draw(output='text')))
        
    # Load results
    with open(res_file, 'r') as f:
        res = json.load(f, cls=RuntimeDecoder)
        
    counts = res[0].data.meas.get_counts()
    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:15])
    
    # Draw histogram
    hist_path = r"d:\Cosmos\workloads (2)\histogram_d6lmi8u.png"
    fig = plot_histogram(sorted_counts, title="Swarm 54D Quantum Measurement Outcomes", color='#1f77b4', figsize=(12,6))
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    print("SUCCESS")

if __name__ == "__main__":
    generate_graphs()
