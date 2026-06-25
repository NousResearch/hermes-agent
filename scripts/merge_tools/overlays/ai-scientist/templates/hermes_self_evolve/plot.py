import json
import os
import os.path as osp

import matplotlib.pyplot as plt

for folder in [f for f in os.listdir("./") if f.startswith("run") and osp.isdir(f)]:
    with open(osp.join(folder, "final_info.json"), "r", encoding="utf-8") as handle:
        means = json.load(handle)["hermes_self_evolve"]["means"]
    keys = list(means.keys())
    vals = [means[k] for k in keys]
    plt.figure(figsize=(7, 4))
    plt.bar(keys, vals, color="#B279A2")
    plt.title(f"hermes_self_evolve ({folder})")
    plt.tight_layout()
    plt.savefig(f"hermes_self_evolve_{folder}.png")
    plt.close()
