\# Task drafts (Week 1)

Format: id | split-placeholder | check | command



001 | train | exit-code 0 + stdout equals 4 | python -c "print(2+2)"

002 | train | file content equals "hello" | (create hello.txt with hello)

003 | train | pytest exit code 0 | pytest <fixture>

004 | train | JSON field equals 1 | python -c "import json;print(json.load(open('x.json'))\['a'])"

005 | train | script runs without error | python bad.py

006 | train | pytest passes | pytest <fn fixture>

007 | train | stdout equals row count | python -c "import csv;print(len(list(csv.reader(open('d.csv')))))"

008 | train | target file exists | (rename file)

009 | train | file absent | (delete file)

010 | train | stdout contains pattern | findstr <pat> file

011 | train | directories exist | (mkdir -p a\\b\\c)

012 | train | stdout matches "ok" | (echo ok)

013 | train | stdout equals \[1,2,3] | python -c "print(sorted(\[3,1,2]))"

014 | train | exit code 0 | bash script.sh

015 | train | output is valid CSV | python -c "import csv;list(csv.reader(open('o.csv')))"

016 | train | stdout equals 3 | python -c "print(max(\[3,1,2]))"

017 | train | stdout contains branch name | git rev-parse --abbrev-ref HEAD

018 | train | exit code 0 | ruff check .

019 | train | output file exists and is smaller | gzip -k file

020 | train | file contains '#' | findstr "#" doc.md
# Memory-touching tasks 

021 | train | new session recalls fact from prior session | recall_test
022 | train | preference applied 5 turns after stated | preference_test
023 | train | extraction fires on session end | extract_check
024 | train | retrieval surfaces specific memory | precision_test
025 | train | cross-agent: agent A writes, agent B reads | cross_agent_test
026 | heldout | same as 021, unseen variant | recall_heldout

