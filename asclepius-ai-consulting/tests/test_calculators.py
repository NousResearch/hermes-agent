import importlib.util, unittest
from pathlib import Path
def load(p,n):
    spec=importlib.util.spec_from_file_location(n, Path(__file__).parents[1]/p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
pricing=load('finance/pricing-calculator.py','pricing')
roi=load('finance/roi-calculator-for-prospects.py','roi')
class CalculatorTests(unittest.TestCase):
    def test_build_price_known_input(self): self.assertEqual(pricing.calculate('build',20,1.5),(4500,225))
    def test_diagnostic_floor(self): self.assertEqual(pricing.calculate('diagnostic',1,1),(999,999))
    def test_roi(self): self.assertEqual(roi.calculate(6,150),(46800,46.8))
if __name__=='__main__': unittest.main()
