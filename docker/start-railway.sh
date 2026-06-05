#!/bin/bash
# Script de démarrage pour Railway

echo "🚀 Démarrage de Hermes-Agent sur Railway..."

# 1. Lancer la Gateway en arrière-plan
# La gateway est le "cerveau" qui fait tourner Claude
hermes gateway run &
GATEWAY_PID=$!

# Attendre quelques secondes que la gateway s'initialise
sleep 5

# 2. Lancer le Dashboard (Interface Web) au premier plan
# Railway injecte la variable $PORT. S'il n'y en a pas, on utilise 9119 par défaut.
PORT="${PORT:-9119}"
echo "🌐 Lémarrage du Dashboard web sur le port $PORT..."
hermes dashboard --host 0.0.0.0 --port $PORT --no-open

# Si le dashboard s'arrête, on tue aussi la gateway
kill $GATEWAY_PID
