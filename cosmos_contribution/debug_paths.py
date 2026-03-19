
import sys
import os

try:
    import cosmos
    print(f"cosmos location: {cosmos.__file__}")
except ImportError:
    print("cosmos not found")

try:
    import cosmosynapse
    print(f"cosmosynapse location: {cosmosynapse.__file__}")
except ImportError:
    print("cosmosynapse not found")

print("\nsys.path:")
for p in sys.path:
    print(f"  {p}")
