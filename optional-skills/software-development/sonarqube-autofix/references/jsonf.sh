#!/usr/bin/env bash
set -euo pipefail

# Script to format JSON with indentation
# Usage: jsonf <file.json>

jsonf(){
  f=$1
  i=0
  s=false
  e=false

  while IFS= read -r -n1 c; do
    if $s; then
      if $e; then
        printf '%s' "$c"
        e=false
      elif [[ $c == '\\' ]]; then
        printf '%s' "$c"
        e=true
      elif [[ $c == '"' ]]; then
        printf '%s' "$c"
        s=false
      else
        printf '%s' "$c"
      fi
    else
      case $c in
        "{"|"[")
          printf '%s' "$c"
          echo
          ((i+=4))
          printf '%*s' $i ''
          ;;
        "}"|"]")
          echo
          ((i-=4))
          printf '%*s' $i ''
          printf '%s' "$c"
          ;;
        ",")
          printf '%s\n' "$c"
          printf '%*s' $i ''
          ;;
        ":")
          printf '%s' ": "
          ;;
        '"')
          printf '%s' "$c"
          s=true
          ;;
        *)
          if [[ ! $c =~ [[:space:]] ]]; then
            printf '%s' "$c"
          fi
          ;;
      esac
    fi
  done < "$f"
}

jsonf "$1" | sed '/^[[:space:]]\+$/d'
