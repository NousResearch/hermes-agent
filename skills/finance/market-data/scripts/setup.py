python
    #!/usr/bin/env python3
    import sys
    import subprocess
    import os

    def install_deps():
        print("Installing dependencies for finance/market-data...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
            print("Installation complete: yfinance")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            sys.exit(1)

    if __name__ == "__main__":
        if "--check" in sys.argv:
            try:
                import yfinance
                print("INSTALLED")
                sys.exit(0)
            except ImportError:
                print("NOT_INSTALLED")
                sys.exit(1)
        else:
            install_deps()
