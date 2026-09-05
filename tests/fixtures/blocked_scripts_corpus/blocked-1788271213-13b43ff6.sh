# Final verification: root + all profiles
echo "=== root skill + contract ==="
ls "$LOCALAPPDATA/hermes/skills/caveman/SKILL.md" && grep -c "## Caveman default" "$LOCALAPPDATA/hermes/AGENTS.md"
echo "=== profile skills ==="
for d in "$LOCALAPPDATA/hermes/profiles/"*/skills/caveman/SKILL.md; do echo "OK $(echo $d | grep -o 'profiles/[a-z-]*')"; done | wc -l
echo "=== profile contracts ==="
grep -l "## Caveman default" "$LOCALAPPDATA/hermes/profiles/"*/AGENTS.md | wc -l
