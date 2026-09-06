cd "$HOME/AppData/Local/hermes"
python -c "import ruamel.yaml; print('ruamel OK', ruamel.yaml.__version__)" 2>&1 | head -1
echo "--- confirm hypermind provider string form used at runtime (from logs) ---"
grep -ohE "provider=custom[^ ]*|custom:hypermind[^ ,'\"]*" logs/agent.log 2>/dev/null | sort -u | head
