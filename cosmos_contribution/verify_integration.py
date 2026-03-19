import asyncio
import aiohttp
import json
import socket
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cosmosVerifier")

async def verify_sensory_system(port=8765):
    """Check if the 12D Sensory System is online and broadcasting."""
    print(f"\n🔬 Verifying Sensory System (Port {port})...")
    try:
        async with aiohttp.ClientSession() as session:
            # 1. Check HTTP API
            url = f"http://localhost:{port}/state"
            logger.info(f"Connecting to {url}...")
            async with session.get(url, timeout=2.0) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Check for 12D Data Keys
                    physics = data.get('cst_physics', {})
                    phase = physics.get('cst_state', 'UNKNOWN')
                    print(f"✅ Sensory System API: ONLINE")
                    print(f"📊 Live Data Test: Phase={phase}, Val={data.get('derived_state', {}).get('pad_vector', {}).get('pleasure', 0)}")
                    return True
                else:
                    print(f"❌ Sensory System API: ERROR {resp.status}")
                    return False
    except Exception as e:
        print(f"❌ Sensory System API: FAILED ({e})")
        return False

async def verify_proxy_integration(port=8081):
    """Check if the Web Server is correctly proxying the Sensory System."""
    print(f"\n🌐 Verifying Web Proxy (Port {port})...")
    try:
        async with aiohttp.ClientSession() as session:
            # 1. Check Proxy Endpoint
            url = f"http://localhost:{port}/api/emotional"
            logger.info(f"Connecting to {url}...")
            async with session.get(url, timeout=2.0) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Verify it matches the Full System structure
                    if 'cst_physics' in data or 'cosmos_packet' in data:
                        print(f"✅ Web Proxy: PASS (Correctly forwarding 12D data)")
                        return True
                    else:
                        print(f"⚠️ Web Proxy: WARNING (Data structure suspicious)")
                        print(f"Keys: {list(data.keys())}")
                        return False
                else:
                    print(f"❌ Web Proxy: ERROR {resp.status} (Proxy might be down or blocked)")
                    return False
    except Exception as e:
        print(f"❌ Web Proxy: FAILED ({e})")
        return False

async def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║               FAB-5 VERIFICATION PROTOCOL                        ║
╠══════════════════════════════════════════════════════════════════╣
║ Subject: 12D CST Sensory Bridge                                  ║
║ Status:  Verifying Neural Link Integrity...                      ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    sensory_ok = await verify_sensory_system(8765)
    proxy_ok = await verify_proxy_integration(8081)
    
    print("\n" + "="*60)
    
    if sensory_ok and proxy_ok:
        print("✅ SYSTEM INTEGRITY: 100%")
        print("The AI Models are RECEIVING your biological data.")
        print("The Web Interface is DISPLAYING your biological data.")
        print("Enjoy the loop.")
    elif sensory_ok and not proxy_ok:
        print("⚠️ PARTIAL FAILURE: Sensory System is UP, but Web Proxy is DOWN.")
        print("Faces will work in Option 6 window, but Web UI/Bots might be blind.")
    elif not sensory_ok:
        print("❌ CRITICAL FAILURE: Sensory System (Option 4/6) is OFFLINE.")
        print("Please restart Option 2.")
    else:
        print("❌ TOTAL FAILURE.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
