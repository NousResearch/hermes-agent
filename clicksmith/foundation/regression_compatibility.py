"""FND-09 — Regression / Compatibility Contract v1.1."""
from __future__ import annotations
import dataclasses, json, re
from typing import Any
from .execution_identity import ExecutionIdentity, ExecutionIdentityError

SCHEMA_VERSION = "1.1"
REGRESSION_RESULTS = frozenset({"REGRESSION_FREE", "REGRESSED", "INCONCLUSIVE"})
COMPATIBILITY_RESULTS = frozenset({"COMPATIBLE", "INCOMPATIBLE", "INCONCLUSIVE"})
APPLICABILITY = frozenset({"APPLICABLE", "NOT_APPLICABLE"})
COMPATIBILITY_DIMENSIONS = frozenset({"API", "SCHEMA", "BEHAVIOR", "PERFORMANCE", "DEPENDENCY", "RUNTIME", "SECURITY", "CONFIGURATION"})
COMPARISON_METHODS = frozenset({"EXPLICIT_EVIDENCE", "EXPLICIT_DETERMINATION"})
ERROR_CODES = frozenset({
    "INVALID_IDENTITY", "INVALID_REGRESSION_INPUT", "INVALID_COMPATIBILITY_INPUT",
    "INVALID_BASELINE", "INVALID_COMPARISON_SCOPE", "INVALID_CRITERION",
    "INVALID_COMPATIBILITY_REQUIREMENT", "AMBIGUOUS_REGRESSION", "AMBIGUOUS_COMPATIBILITY",
    "UNSUPPORTED_COMPATIBILITY_DIMENSION", "UNSUPPORTED_COMPARISON_METHOD",
    "REGRESSION_EVALUATION_ERROR", "COMPATIBILITY_EVALUATION_ERROR",
    "REGRESSION_BOUNDARY_VIOLATION", "COMPATIBILITY_BOUNDARY_VIOLATION", "SCHEMA_ERROR",
})
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")

def _id(name: str, value: Any, exc: type[Exception]) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value): raise exc(f"{name} must be a canonical identifier.")
def _text(name: str, value: Any, exc: type[Exception]) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096: raise exc(f"{name} must be a bounded non-empty string.")
def _json(value: Any) -> str:
    try: return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as e: raise SchemaError("canonical serialization failed.") from e

class RegressionCompatibilityError(ValueError): code = "SCHEMA_ERROR"
class InvalidIdentityError(RegressionCompatibilityError): code = "INVALID_IDENTITY"
class InvalidRegressionInputError(RegressionCompatibilityError): code = "INVALID_REGRESSION_INPUT"
class InvalidCompatibilityInputError(RegressionCompatibilityError): code = "INVALID_COMPATIBILITY_INPUT"
class InvalidBaselineError(RegressionCompatibilityError): code = "INVALID_BASELINE"
class InvalidComparisonScopeError(RegressionCompatibilityError): code = "INVALID_COMPARISON_SCOPE"
class InvalidCriterionError(RegressionCompatibilityError): code = "INVALID_CRITERION"
class InvalidCompatibilityRequirementError(RegressionCompatibilityError): code = "INVALID_COMPATIBILITY_REQUIREMENT"
class AmbiguousRegressionError(RegressionCompatibilityError): code = "AMBIGUOUS_REGRESSION"
class AmbiguousCompatibilityError(RegressionCompatibilityError): code = "AMBIGUOUS_COMPATIBILITY"
class UnsupportedCompatibilityDimensionError(RegressionCompatibilityError): code = "UNSUPPORTED_COMPATIBILITY_DIMENSION"
class UnsupportedComparisonMethodError(RegressionCompatibilityError): code = "UNSUPPORTED_COMPARISON_METHOD"
class RegressionEvaluationError(RegressionCompatibilityError): code = "REGRESSION_EVALUATION_ERROR"
class CompatibilityEvaluationError(RegressionCompatibilityError): code = "COMPATIBILITY_EVALUATION_ERROR"
class RegressionBoundaryViolationError(RegressionCompatibilityError): code = "REGRESSION_BOUNDARY_VIOLATION"
class CompatibilityBoundaryViolationError(RegressionCompatibilityError): code = "COMPATIBILITY_BOUNDARY_VIOLATION"
class SchemaError(RegressionCompatibilityError): code = "SCHEMA_ERROR"

@dataclasses.dataclass(frozen=True)
class BaselineReference:
    baseline_id: str; subject: str; version_or_revision: str; comparison_scope_id: str
    def validate(self):
        for n,v in dataclasses.asdict(self).items(): _id(n,v,InvalidBaselineError)
    def to_dict(self): self.validate(); return dataclasses.asdict(self)

@dataclasses.dataclass(frozen=True)
class ChangeReference:
    change_id: str; subject: str; version_or_revision: str; change_description: str
    def validate(self):
        _id("change_id",self.change_id,InvalidRegressionInputError); _text("subject",self.subject,InvalidRegressionInputError); _text("version_or_revision",self.version_or_revision,InvalidRegressionInputError); _text("change_description",self.change_description,InvalidRegressionInputError)
    def to_dict(self): self.validate(); return dataclasses.asdict(self)

@dataclasses.dataclass(frozen=True)
class ComparisonScope:
    scope_id: str; subject: str; declared_dimensions: tuple[str,...]=(); declared_criteria: tuple[str,...]=()
    def validate(self):
        _id("scope_id",self.scope_id,InvalidComparisonScopeError); _text("subject",self.subject,InvalidComparisonScopeError)
        if not self.declared_dimensions and not self.declared_criteria: raise InvalidComparisonScopeError("comparison scope must be explicit.")
        if any(not isinstance(x,str) or not x.strip() for x in (*self.declared_dimensions,*self.declared_criteria)): raise InvalidComparisonScopeError("scope entries must be explicit strings.")
    def to_dict(self): self.validate(); return {"scope_id":self.scope_id,"subject":self.subject,"declared_dimensions":list(self.declared_dimensions),"declared_criteria":list(self.declared_criteria)}

@dataclasses.dataclass(frozen=True)
class RegressionCriterion:
    criterion_id: str; description: str; required: bool; comparison_method: str
    def validate(self):
        _id("criterion_id",self.criterion_id,InvalidCriterionError); _text("description",self.description,InvalidCriterionError)
        if type(self.required) is not bool: raise InvalidCriterionError("required must be boolean.")
        if self.comparison_method not in COMPARISON_METHODS: raise UnsupportedComparisonMethodError("comparison method is unsupported.")
    def to_dict(self): self.validate(); return dataclasses.asdict(self)

@dataclasses.dataclass(frozen=True)
class CompatibilityDimension:
    dimension_id: str; description: str; applicability: str
    def validate(self):
        _id("dimension_id",self.dimension_id,InvalidCompatibilityInputError); _text("description",self.description,InvalidCompatibilityInputError)
        if self.dimension_id not in COMPATIBILITY_DIMENSIONS: raise UnsupportedCompatibilityDimensionError("compatibility dimension is unsupported.")
        if self.applicability not in APPLICABILITY: raise InvalidCompatibilityInputError("applicability is invalid.")
    def to_dict(self): self.validate(); return dataclasses.asdict(self)

@dataclasses.dataclass(frozen=True)
class CompatibilityRequirement:
    requirement_id: str; description: str; required: bool; dimension_id: str
    def validate(self):
        _id("requirement_id",self.requirement_id,InvalidCompatibilityRequirementError); _text("description",self.description,InvalidCompatibilityRequirementError)
        if type(self.required) is not bool: raise InvalidCompatibilityRequirementError("required must be boolean.")
        _id("dimension_id",self.dimension_id,InvalidCompatibilityRequirementError)
    def to_dict(self): self.validate(); return dataclasses.asdict(self)

@dataclasses.dataclass(frozen=True)
class RegressionAssessment:
    regression_id: str; baseline_reference: BaselineReference; change_reference: ChangeReference
    comparison_scope: ComparisonScope; criteria: tuple[RegressionCriterion,...]; result: str
    execution_identity: ExecutionIdentity|None = None; metadata_or_reference: str|None = None
    def validate(self):
        _id("regression_id",self.regression_id,InvalidRegressionInputError); self.baseline_reference.validate(); self.change_reference.validate(); self.comparison_scope.validate()
        if self.baseline_reference.comparison_scope_id != self.comparison_scope.scope_id: raise AmbiguousRegressionError("baseline and scope do not match.")
        if not self.criteria: raise InvalidRegressionInputError("criteria must be explicit.")
        for c in self.criteria: c.validate()
        if self.result not in REGRESSION_RESULTS: raise InvalidRegressionInputError("invalid regression result.")
        _validate_identity(self.execution_identity)
    def to_dict(self):
        self.validate(); d={"regression_id":self.regression_id,"baseline_reference":self.baseline_reference.to_dict(),"change_reference":self.change_reference.to_dict(),"comparison_scope":self.comparison_scope.to_dict(),"criteria":[c.to_dict() for c in self.criteria],"result":self.result,"metadata_or_reference":self.metadata_or_reference}
        if self.execution_identity: d["execution_identity"]=self.execution_identity.to_dict()
        return d
    def serialize(self): return _json(self.to_dict())

@dataclasses.dataclass(frozen=True)
class CompatibilityAssessment:
    compatibility_id: str; comparison_reference: ChangeReference; comparison_scope: ComparisonScope
    requirements: tuple[CompatibilityRequirement,...]; dimensions: tuple[CompatibilityDimension,...]; result: str
    execution_identity: ExecutionIdentity|None = None; metadata_or_reference: str|None = None
    def validate(self):
        _id("compatibility_id",self.compatibility_id,InvalidCompatibilityInputError); self.comparison_reference.validate(); self.comparison_scope.validate()
        if not self.requirements: raise InvalidCompatibilityInputError("requirements must be explicit.")
        if not self.dimensions: raise InvalidCompatibilityInputError("dimensions must be explicit.")
        for d in self.dimensions: d.validate()
        for r in self.requirements: r.validate()
        if self.result not in COMPATIBILITY_RESULTS: raise InvalidCompatibilityInputError("invalid compatibility result.")
        _validate_identity(self.execution_identity)
    def to_dict(self):
        self.validate(); d={"compatibility_id":self.compatibility_id,"comparison_reference":self.comparison_reference.to_dict(),"comparison_scope":self.comparison_scope.to_dict(),"requirements":[r.to_dict() for r in self.requirements],"dimensions":[x.to_dict() for x in self.dimensions],"result":self.result,"metadata_or_reference":self.metadata_or_reference}
        if self.execution_identity: d["execution_identity"]=self.execution_identity.to_dict()
        return d
    def serialize(self): return _json(self.to_dict())

def _validate_identity(identity):
    if identity is None: return
    if not isinstance(identity,ExecutionIdentity): raise InvalidIdentityError("execution_identity must be an ExecutionIdentity.")
    try: identity.validate()
    except ExecutionIdentityError as e: raise InvalidIdentityError(str(e)) from e

def _same_identity(a,b):
    if a is None or b is None: return a is b
    return a.task_id==b.task_id and a.run_id==b.run_id and a.attempt_id==b.attempt_id and a.revision==b.revision

def evaluate_regression(*, regression_id, baseline, change, scope, criteria, determination=None, execution_identity=None, metadata_or_reference=None):
    if baseline is None: raise InvalidBaselineError("baseline is required and must be explicit.")
    if not isinstance(baseline,BaselineReference): raise InvalidBaselineError("baseline must be a BaselineReference.")
    if not isinstance(change,ChangeReference): raise InvalidRegressionInputError("change must be a ChangeReference.")
    if not isinstance(scope,ComparisonScope): raise InvalidComparisonScopeError("scope must be a ComparisonScope.")
    if determination is not None and determination not in REGRESSION_RESULTS: raise InvalidRegressionInputError("invalid regression determination.")
    if determination is None: determination="INCONCLUSIVE"
    assessment=RegressionAssessment(regression_id,baseline,change,scope,tuple(criteria),determination,execution_identity,metadata_or_reference)
    assessment.validate(); return assessment

def evaluate_compatibility(*, compatibility_id, comparison_reference, scope, requirements, dimensions, determination=None, execution_identity=None, metadata_or_reference=None):
    if comparison_reference is None: raise InvalidCompatibilityInputError("comparison reference is required.")
    if not isinstance(comparison_reference,ChangeReference): raise InvalidCompatibilityInputError("comparison reference is invalid.")
    if not isinstance(scope,ComparisonScope): raise InvalidComparisonScopeError("scope must be a ComparisonScope.")
    if determination is not None and determination not in COMPATIBILITY_RESULTS: raise InvalidCompatibilityInputError("invalid compatibility determination.")
    for d in dimensions:
        if not isinstance(d,CompatibilityDimension): raise InvalidCompatibilityInputError("dimension is invalid.")
    result=determination or "INCONCLUSIVE"
    assessment=CompatibilityAssessment(compatibility_id,comparison_reference,scope,tuple(requirements),tuple(dimensions),result,execution_identity,metadata_or_reference)
    assessment.validate(); return assessment
