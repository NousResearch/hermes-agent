import pytest
from clicksmith.foundation.execution_identity import ExecutionIdentity
from clicksmith.foundation.regression_compatibility import *

BASE=BaselineReference("base-1","app","v1","scope-1")
CHANGE=ChangeReference("change-1","app","v2","changed")
SCOPE=ComparisonScope("scope-1","app",(),("crit-1",))
CRIT=RegressionCriterion("crit-1","behavior",True,"EXPLICIT_DETERMINATION")
DIM=CompatibilityDimension("API","API contract","APPLICABLE")
REQ=CompatibilityRequirement("req-1","API compatible",True,"API")
ID=ExecutionIdentity("task-1","run-1","attempt-1",None,1)

def reg(**kw):
    args=dict(regression_id="reg-1",baseline=BASE,change=CHANGE,scope=SCOPE,criteria=(CRIT,))
    args.update(kw); return evaluate_regression(**args)
def comp(**kw): return evaluate_compatibility(compatibility_id="cmp-1",comparison_reference=CHANGE,scope=SCOPE,requirements=(REQ,),dimensions=(DIM,),**kw)

def test_01_missing_baseline():
    with pytest.raises(InvalidBaselineError): reg(baseline=None)
def test_02_ambiguous_baseline():
    with pytest.raises(InvalidBaselineError): BaselineReference("bad id","app","v1","scope-1").validate()
def test_03_missing_scope():
    with pytest.raises(InvalidComparisonScopeError): reg(scope=ComparisonScope("s","app"))
def test_04_ambiguous_scope():
    with pytest.raises(AmbiguousRegressionError): reg(scope=ComparisonScope("other","app",(),("crit-1",)))
def test_05_unsupported_method():
    with pytest.raises(UnsupportedComparisonMethodError): RegressionCriterion("c","x",True,"RUNNER").validate()
def test_06_unsupported_dimension():
    with pytest.raises(UnsupportedCompatibilityDimensionError): CompatibilityDimension("UNKNOWN","x","APPLICABLE").validate()
def test_07_non_applicable_dimension():
    d=CompatibilityDimension("API","API","NOT_APPLICABLE"); d.validate(); assert d.applicability=="NOT_APPLICABLE"
def test_08_identity_mismatch():
    bad=ExecutionIdentity("task-2","run-1","attempt-1",None,1)
    a=reg(execution_identity=ID); b=reg(execution_identity=bad); assert a.execution_identity!=b.execution_identity

def test_09_cross_attempt_not_silently_reinterpreted():
    a=reg(execution_identity=ID); b=reg(execution_identity=ExecutionIdentity("task-1","run-1","attempt-2",None,1)); assert a.execution_identity.attempt_id!=b.execution_identity.attempt_id

def test_10_external_failure_not_regression():
    assert reg(determination="INCONCLUSIVE").result=="INCONCLUSIVE"
def test_11_inconclusive_not_free(): assert reg(determination="INCONCLUSIVE").result!="REGRESSION_FREE"
def test_12_inconclusive_not_regressed(): assert reg(determination="INCONCLUSIVE").result!="REGRESSED"
def test_13_inconclusive_not_compatible(): assert comp(determination="INCONCLUSIVE").result!="COMPATIBLE"
def test_14_inconclusive_not_incompatible(): assert comp(determination="INCONCLUSIVE").result!="INCOMPATIBLE"
def test_15_baseline_mutation_rejected():
    with pytest.raises(Exception): BASE.baseline_id="x"
def test_16_no_implicit_baseline():
    with pytest.raises(InvalidBaselineError): evaluate_regression(regression_id="r",baseline=None,change=CHANGE,scope=SCOPE,criteria=(CRIT,))
def test_17_scope_must_be_explicit():
    with pytest.raises(InvalidComparisonScopeError): ComparisonScope("s","app").validate()
def test_18_invalid_criterion():
    with pytest.raises(InvalidCriterionError): RegressionCriterion("","x",True,"EXPLICIT_DETERMINATION").validate()
def test_19_invalid_requirement():
    with pytest.raises(InvalidCompatibilityRequirementError): CompatibilityRequirement("","x",True,"API").validate()
def test_20_invalid_identity():
    with pytest.raises(InvalidIdentityError): reg(execution_identity="not-identity").validate()
def test_21_semantic_failure_is_not_external_failure():
    with pytest.raises(InvalidRegressionInputError): reg(determination="BROKEN")
def test_22_assessment_id_omission_rejected():
    with pytest.raises(TypeError): evaluate_regression(baseline=BASE,change=CHANGE,scope=SCOPE,criteria=(CRIT,))
def test_23_boundary_violation():
    with pytest.raises(InvalidIdentityError): reg(execution_identity=object()).validate()
def test_24_success_does_not_authorize():
    assert reg(determination="REGRESSION_FREE").result=="REGRESSION_FREE"
def test_25_success_not_production_ready():
    assert comp(determination="COMPATIBLE").result=="COMPATIBLE"

def test_result_vocabularies_exact():
    assert REGRESSION_RESULTS=={"REGRESSION_FREE","REGRESSED","INCONCLUSIVE"}
    assert COMPATIBILITY_RESULTS=={"COMPATIBLE","INCOMPATIBLE","INCONCLUSIVE"}

def test_error_vocabularies_exact():
    assert len(ERROR_CODES)==16

def test_deterministic_serialization():
    a=reg(determination="REGRESSION_FREE").serialize(); b=reg(determination="REGRESSION_FREE").serialize(); assert a==b

def test_compatibility_known_dimension_validation():
    DIM.validate(); assert DIM.dimension_id=="API"

def test_compatibility_inconclusive_default():
    assert comp().result=="INCONCLUSIVE"

def test_regression_inconclusive_default():
    assert reg().result=="INCONCLUSIVE"

def test_execution_identity_preserved():
    assert reg(execution_identity=ID).execution_identity==ID

def test_models_are_immutable():
    with pytest.raises(Exception): BASE.baseline_id="mutated"
