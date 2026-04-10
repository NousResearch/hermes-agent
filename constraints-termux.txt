# Termux / Android dependency constraints for Hermes Agent.
#
# Usage:
#   python -m pip install -e '.[termux]' -c constraints-termux.txt
#
# These pins keep the tested Android install path stable when upstream packages
# move faster than Termux-compatible wheels / sdists.

ipython<10
jedi>=0.18.1,<0.20
parso>=0.8.4,<0.9
stack-data>=0.6,<0.7
pexpect>4.3,<5
matplotlib-inline>=0.1.7,<0.2
asttokens>=2.1,<3
