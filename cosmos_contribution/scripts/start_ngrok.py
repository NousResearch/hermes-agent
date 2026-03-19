import os
import time
from pyngrok import ngrok
from dotenv import load_dotenv

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root_dir, '.env')
    load_dotenv(env_path)
    
    auth_token = os.getenv("NGROK_AUTH_TOKEN")
    
    if not auth_token:
        print("Ngrok requires a free Auth Token to host web traffic to the internet.")
        print("You can get your free token by signing up at: https://dashboard.ngrok.com/get-started/your-authtoken")
        auth_token = input("\nPaste your NGROK_AUTH_TOKEN here: ").strip()
        if auth_token:
            ngrok.set_auth_token(auth_token)
            # Save to .env for next time
            if not os.path.exists(env_path):
                open(env_path, 'a').close()
            with open(env_path, 'a') as f:
                f.write(f"\nNGROK_AUTH_TOKEN={auth_token}\n")
            print("Token securely saved to .env!")
        else:
            print("\n⚠️ No token provided. Ngrok may reject the connection.")
            time.sleep(2)
    else:
        ngrok.set_auth_token(auth_token)
        
    print("\nStarting Ngrok tunnel for Cosmos Web Server (Port 8081)...")
    try:
        # Create tunnel
        public_url = ngrok.connect(8081, bind_tls=True)
        print("\n" + "="*50)
        print(" 🚀 NGROK PUBLIC TUNNEL ACTIVE 🚀")
        print("="*50)
        print(f"\n Your Global Cosmos URL is:  ")
        print(f" >>> {public_url.public_url} <<<")
        print("\n You can now send this link to anyone, or use it on your phone over 5G/LTE.")
        print(" NOTE: Make sure 'Option 2' (Cosmos Web Server) is currently running in another terminal!")
        print("\n Press [Ctrl+C] to close the internet tunnel.")
        
        # Keep the process alive
        ngrok_process = ngrok.get_ngrok_process()
        ngrok_process.proc.wait()
    except KeyboardInterrupt:
        print("\nClosing Ngrok tunnel...")
        ngrok.kill()
    except Exception as e:
        print(f"\nError starting Ngrok: {e}")

if __name__ == "__main__":
    main()
