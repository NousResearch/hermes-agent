import ast
import re

with open('gateway/run.py', 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

tree = ast.parse('\n'.join(lines))
replacements = []

for node in ast.walk(tree):
    if isinstance(node, ast.ExceptHandler):
        # Is the body just a single 'pass' node?
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            pass_node = node.body[0]
            lineno = pass_node.lineno - 1  # 0-indexed
            col_offset = pass_node.col_offset
            
            # Find the exception type if any
            exc_type = 'Exception'
            if node.type:
                if isinstance(node.type, ast.Name):
                    exc_type = node.type.id
                elif isinstance(node.type, ast.Attribute):
                    exc_type = node.type.value.id + '.' + node.type.attr
            
            # Find the exception name if 'as e'
            if node.name:
                exc_var = node.name
            else:
                exc_var = '_err'
            
            replacements.append((lineno, col_offset, exc_type, exc_var, node.type is None, node.name is None))

print('Found', len(replacements), 'empty pass handlers.')

# We apply modifications backwards so line changes (if any) don't affect previous lines
for r in reversed(sorted(replacements, key=lambda x: x[0])):
    lineno, col, exc_type, exc_var, no_type, no_name = r
    indent = ' ' * col
    
    # We want to replace the `except [Exception]:\n    pass`
    # Let's locate the except line.
    except_lineno = lineno
    while except_lineno > 0 and 'except' not in lines[except_lineno]:
        except_lineno -= 1
        
    line_except = lines[except_lineno]
    
    # Let's rewrite the except line to ensure it has 'as _err' if needed
    if no_name and not no_type:
        # Match 'except ValueError:' and replace with 'except ValueError as _err:'
        # Wait, there could be multiple exceptions like 'except (ValueError, OSError):'
        # It's safer to just regex replace the colon
        new_except = re.sub(r':\s*$', f' as {exc_var}:', line_except)
    elif no_type:
        # It's 'except:'
        new_except = re.sub(r'except\s*:\s*$', f'except Exception as {exc_var}:', line_except)
    else:
        new_except = line_except
        
    lines[except_lineno] = new_except
    
    # Replace the pass
    lines[lineno] = f'{indent}logger.warning(f\"Ignored {exc_type}: {{{exc_var}}}\")'

with open('gateway/run.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
