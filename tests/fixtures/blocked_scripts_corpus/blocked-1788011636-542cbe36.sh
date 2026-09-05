cd "$HOME/AppData/Local/hermes"
if ! grep -q '^HYPERMIND_API_KEY=' .env; then
  KEY=$(python -c "import yaml;print(yaml.safe_load(open('config.yaml'))['custom_providers'][0]['api_key'])")
  printf '\n# Hypermind.app (moved from config.yaml plaintext for security)\nHYPERMIND_API_KEY=%s\n' "$KEY" >> .env
  echo "RESULT: appended HYPERMIND_API_KEY to .env"
else
  echo "RESULT: HYPERMIND_API_KEY already present"
fi
echo "present count: $(grep -c '^HYPERMIND_API_KEY=' .env)"
