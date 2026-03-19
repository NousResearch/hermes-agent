# 📱 Host Cosmos on Your Phone (Local Network)

Cosmos is designed to run its heavy AI models (Ollama, Gemini, Claude) on your PC, while letting you interact with it from your phone like a native app. 

Because `run_web.py` binds to `0.0.0.0` automatically, the server is already broadcast across your local Wi-Fi out of the box!

## 🚀 How to Connect Your Phone

1. **Start Cosmos on your PC:**
   Double-click `START.bat` and select **Option 13** (Full System). Wait for the Web Interface to say "running on http://0.0.0.0:8081".

2. **Find Your PC's Local IP:**
   Double-click the included `show_ip_for_phone.bat` script.
   *It will print something like `http://192.168.1.55:8081`.*

3. **Open on your Phone:**
   Make sure your phone is connected to the **same Wi-Fi network** as your PC. 
   Open Safari or Chrome and type in the exact URL from step 2.

4. **Enjoy the Mobile UI:**
   The interface has been specifically optimized with CSS media queries to look and feel like a native mobile app on your phone.

---

## 🔑 Running Your Own Models (API Setup)
As a developer using Cosmos, you have full control over the AI swarm. There are NO limits (the 2000 character limit and token restrictions have been permanently removed).

To unlock the cloud APIs (Gemini, ChatGPT, etc.), simply create a `.env` file in the root directory (you can copy `.env.example`) and add your keys:

```ini
# Add to .env file
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
IBM_QUANTUM_TOKEN=your_token_here
```

**Local Models (Ollama):**
By default, Cosmos relies on Ollama for ultimate privacy. The default model is `llama3.1:8b`. If you want to use a different local model, add this to your `.env`:
```ini
cosmos_PRIMARY_MODEL=qwen2.5:14b
```

---

## 🛡️ Troubleshooting Phone Connection
If your phone says "Cannot connect to server":
1. **Windows Firewall:** Your PC firewall is blocking the port. Open Start -> search `"Windows Defender Firewall"` -> click `"Allow an app or feature through..."` -> ensure Python is checked for both Private and Public networks.
2. **Same Network:** Double-check that your PC isn't on Ethernet while the phone is on a Guest Wi-Fi network. They must see each other.
