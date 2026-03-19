"""Quick dry-run test for CosmosCNS with Swarm Plasticity integration."""
import logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

from cosmos_cns import CosmosCNS

print()
cns = CosmosCNS()
print()
stats = cns.start_life(dry_run=True, dry_run_ticks=20)
print()
print("=== PLASTICITY STATS ===")
p = stats.get("plasticity", {})
print("Total updates:", p.get("total_updates", 0))
print("Total blocked:", p.get("total_blocked", 0))
for ctx, w in p.get("weights", {}).items():
    print(f"  {ctx}: {w}")
print("=== DONE ===")
